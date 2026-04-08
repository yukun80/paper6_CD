"""城市洪水拓扑重连图路由模块 (Urban Flood Topological Routing Module)

水是流体，在城市街道中必然是物理连通的，只是在 SAR 图像上被高楼阴影
"视觉切断"了。本模块不用形态学去平滑断裂的洪水预测，而是借鉴 SAM-Road
提取断裂路网的思想，用图网络去"连接"被遮挡切断的水网。

三步核心机制:
1. 网格节点提取 —— 从 Detector 的对比度感知 P2 特征中提取图节点
2. 拓扑边 Transformer —— 利用对比度感知变化特征预测节点间物理连通性
3. 图路由双向精修 —— 沿连通路径传播洪水证据（增强漏检 + 抑制误报）

与前一版的关键改进:
- 图节点特征来自 Detector 输出（已含 ContrastAwareDiff 信号 + Transformer
  全局上下文），而非从 Encoder 原始特征重新计算，实现三模块协同
- 消息传递和空间渲染均支持双向校正（tanh），可同时增强漏检和抑制误报
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TopoEdgeTransformer(nn.Module):
    """边中心图 Transformer，借鉴自 SAM-Road TopoNet。

    对每个源节点的 K 个候选邻居构成的边序列做 self-attention，
    输出每条边的连通性 logit。与 SAM-Road 的区别在于边特征包含
    双时相变化信息而非单时相视觉特征。
    """

    _CHUNK_SIZE = 2048

    def __init__(self, proj_dim: int, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pair_proj = nn.Linear(2 * proj_dim + 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.output_proj = nn.Linear(hidden_dim, 1)

    def _transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """包装 transformer 以支持梯度检查点。"""
        return self.transformer(x)

    def forward(self, node_feat: torch.Tensor, node_pos: torch.Tensor,
                knn_idx: torch.Tensor):
        """
        Parameters
        ----------
        node_feat : [B, N, D]  每个节点的变化特征
        node_pos  : [N, 2]     网格节点归一化坐标 (预计算 buffer)
        knn_idx   : [N, K]     每个节点的 K 近邻索引 (预计算 buffer)

        Returns
        -------
        edge_logits : [B, N, K]  每条边的连通性 logit
        edge_scores : [B, N, K]  sigmoid 后的连通性分数
        """
        B, N, D = node_feat.shape
        K = knn_idx.shape[1]

        knn_flat = knn_idx.view(-1)                            # [N*K]
        src_idx = torch.arange(N, device=node_feat.device
                               ).unsqueeze(1).expand(N, K).reshape(-1)

        src_feat = node_feat[:, src_idx]                       # [B, N*K, D]
        tgt_feat = node_feat[:, knn_flat]

        # 预计算空间偏移（不随 batch 变化，避免 expand 占显存）
        offset = (node_pos[knn_flat] - node_pos[src_idx]).unsqueeze(0)  # [1, N*K, 2]

        pair_feat = torch.cat([src_feat, tgt_feat,
                               offset.expand(B, -1, -1)], dim=-1)
        pair_feat = F.relu(self.pair_proj(pair_feat))          # [B, N*K, H]
        pair_feat = pair_feat.view(B * N, K, -1)

        # 分块 + 梯度检查点处理 transformer，降低峰值显存
        total = pair_feat.shape[0]
        cs = self._CHUNK_SIZE
        if self.training and total > cs:
            chunks = []
            for start in range(0, total, cs):
                chunk = pair_feat[start:start + cs]
                chunks.append(checkpoint(
                    self._transformer_forward, chunk, use_reentrant=False
                ))
            pair_feat = torch.cat(chunks, dim=0)
        else:
            pair_feat = self.transformer(pair_feat)

        pair_feat = pair_feat.view(B, N, K, -1)
        edge_logits = self.output_proj(pair_feat).squeeze(-1)  # [B, N, K]
        edge_scores = torch.sigmoid(edge_logits)

        return edge_logits, edge_scores


class FloodTopoRouter(nn.Module):
    """城市洪水拓扑重连图路由模块。

    接收 Detector 的 P2 特征（已含 ContrastAwareDiff 对比度信号 +
    Transformer 全局上下文）作为图节点特征来源，实现与上游模块的协同。

    Parameters
    ----------
    feat_dim     : Detector P2 特征通道数 (= fpn_channels)
    grid_size    : 节点网格大小 G，产生 G*G 个图节点
    proj_dim     : 节点特征投影维度
    hidden_dim   : TopoEdgeTransformer 隐藏维度
    n_heads      : Transformer 注意力头数
    n_tf_layers  : Transformer 层数
    neighbor_k   : 每节点 KNN 邻居数
    n_hops       : 图消息传递跳数
    line_samples : GT 连通性标签沿连线采样点数
    """

    def __init__(
        self,
        feat_dim: int,
        grid_size: int = 16,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_tf_layers: int = 3,
        neighbor_k: int = 12,
        n_hops: int = 2,
        line_samples: int = 16,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.neighbor_k = neighbor_k
        self.n_hops = n_hops
        self.line_samples = line_samples
        N = grid_size * grid_size

        # ── Step 1：Detector P2 特征投影（已含对比度信号 + 全局上下文） ──
        self.det_feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
        )

        # ── Step 2：拓扑边 Transformer ──
        self.topo_net = TopoEdgeTransformer(
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_tf_layers,
        )

        # ── Step 3：门控精修 ──
        self.hop_alpha = nn.Parameter(torch.full((n_hops,), 0.3))
        # 特征空间消息传递后投影回标量激活
        self.feat_to_act = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(proj_dim // 2, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(2, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
        )
        self.alpha_raw = nn.Parameter(torch.tensor(-3.0))

        # ── 预计算网格坐标与 KNN 索引 ──
        grid_pos, knn_idx = self._build_grid_and_knn(grid_size, neighbor_k)
        self.register_buffer("grid_pos", grid_pos)    # [N, 2]
        self.register_buffer("knn_idx", knn_idx)      # [N, K]

        # GT 连通性标签的线段采样参数 [line_samples]
        self.register_buffer(
            "_line_t", torch.linspace(0.0, 1.0, line_samples)
        )

    # ------------------------------------------------------------------ #
    #  预计算
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_grid_and_knn(G: int, K: int):
        """生成 G×G 规则网格坐标 [0,1]² 并计算每个节点的 K 近邻索引。"""
        coords = torch.linspace(0.5 / G, 1.0 - 0.5 / G, G)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        grid_pos = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [N, 2]
        N = G * G
        K = min(K, N - 1)

        dist = torch.cdist(grid_pos, grid_pos)                       # [N, N]
        # 排除自身（设为大值）再取 top-K 最近
        dist.fill_diagonal_(float("inf"))
        _, knn_idx = dist.topk(K, largest=False)                      # [N, K]
        return grid_pos, knn_idx

    # ------------------------------------------------------------------ #
    #  Step 3 辅助：稀疏 KNN 消息传递（避免稠密 [B,N,N] 邻接矩阵）
    # ------------------------------------------------------------------ #

    def _sparse_message_passing(self, edge_scores: torch.Tensor,
                                node_feat: torch.Tensor,
                                node_act: torch.Tensor) -> torch.Tensor:
        """基于 KNN 稀疏结构的特征空间消息传递，显存 O(N*K) 而非 O(N²)。

        edge_scores : [B, N, K]  每条边的连通性分数
        node_feat   : [B, N, D]  节点变化特征
        node_act    : [B, N]     原始标量洪水概率
        returns     : enhanced_act [B, N]
        """
        B, N, D = node_feat.shape
        K = self.knn_idx.shape[1]
        feat = node_feat

        for hop in range(self.n_hops):
            alpha_h = torch.sigmoid(self.hop_alpha[hop])
            # 取 KNN 邻居特征: [B, N*K, D]
            neighbor_feat = feat[:, self.knn_idx.view(-1)].view(B, N, K, D)
            # 加权聚合（稀疏，仅 K 个邻居）: [B, N, K, D] * [B, N, K, 1] -> sum -> [B, N, D]
            msg = (neighbor_feat * edge_scores.unsqueeze(-1)).sum(dim=2)
            feat = feat + alpha_h * msg

        act_delta = self.feat_to_act(feat).squeeze(-1)                # [B, N]
        enhanced = (node_act + torch.tanh(act_delta)).clamp(0, 1)
        return enhanced

    # ------------------------------------------------------------------ #
    #  拓扑损失
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _compute_gt_connectivity(self, gt_mask: torch.Tensor):
        """从 GT mask 计算网格节点的连通性标签（纯标签，无需梯度）。

        gt_mask : [B, H, W] 或 [B, 1, H, W]，整型或浮点均可
        returns : gt_edge_labels [B, N, K]  float ∈ {0, 1}
        """
        gt_mask = gt_mask.float()
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.unsqueeze(1)   # [B, H, W] -> [B, 1, H, W]
        B = gt_mask.shape[0]
        G = self.grid_size
        N = G * G
        K = self.knn_idx.shape[1]

        gt_grid = F.adaptive_avg_pool2d(gt_mask, (G, G))              # [B, 1, G, G]
        gt_node = gt_grid.view(B, N)                                  # [B, N]

        pos_norm = self.grid_pos * 2.0 - 1.0                         # [N, 2]

        src_idx = torch.arange(N, device=gt_mask.device
                               ).unsqueeze(1).expand(N, K)            # [N, K]
        src_pos = pos_norm[src_idx.reshape(-1)]                       # [N*K, 2]
        tgt_pos = pos_norm[self.knn_idx.reshape(-1)]                  # [N*K, 2]

        t = self._line_t.view(-1, 1, 1)                              # [M, 1, 1]
        sample_pts = src_pos.unsqueeze(0) * (1 - t) + tgt_pos.unsqueeze(0) * t
        M, NK, _ = sample_pts.shape

        sample_pts = sample_pts.view(1, M * NK, 1, 2).expand(B, -1, -1, -1)
        gt_vals = F.grid_sample(
            gt_mask, sample_pts, mode="bilinear",
            padding_mode="zeros", align_corners=False
        )  # [B, 1, M*NK, 1]
        gt_vals = gt_vals.view(B, M, N * K)                          # [B, M, N*K]
        line_avg = gt_vals.mean(dim=1).view(B, N, K)                 # [B, N, K]

        src_flood = (gt_node[:, src_idx.reshape(-1)]).view(B, N, K)
        tgt_flood = (gt_node[:, self.knn_idx.reshape(-1)]).view(B, N, K)

        gt_connected = (
            (src_flood > 0.5).float()
            * (tgt_flood > 0.5).float()
            * (line_avg > 0.5).float()
        )
        return gt_connected

    def _topo_loss(self, edge_logits: torch.Tensor,
                   gt_mask: torch.Tensor) -> torch.Tensor:
        """拓扑连通性监督损失 (BCEWithLogitsLoss)。"""
        gt_labels = self._compute_gt_connectivity(gt_mask)            # [B, N, K]
        loss = F.binary_cross_entropy_with_logits(
            edge_logits, gt_labels, reduction="mean"
        )
        return loss

    # ------------------------------------------------------------------ #
    #  forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        logit_2ch: torch.Tensor,
        det_p2_feat: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        logit_2ch   : [B, 2, H, W]  Detector 输出的初步 2 类 logits
        det_p2_feat : [B, C, h, w]  Detector P2 特征（已含对比度信号 + 全局上下文）
        gt_mask     : [B, 1, H, W]  训练时传入 GT 洪水 mask，推理时为 None

        Returns
        -------
        refined_logit : [B, 2, H, W]  拓扑路由精修后的 2 类 logits
        topo_loss     : scalar tensor 或 None
        """
        B, _, H, W = logit_2ch.shape
        h, w = det_p2_feat.shape[-2:]
        G = self.grid_size

        p_fg = F.softmax(logit_2ch, dim=1)[:, 1:2]                   # [B,1,H,W]

        # ============ Step 1: 网格节点提取 ============
        p_small = F.interpolate(p_fg, size=(h, w),
                                mode="bilinear", align_corners=False)
        node_act = F.adaptive_avg_pool2d(p_small, (G, G))            # [B,1,G,G]
        node_act_flat = node_act.view(B, -1)                         # [B, N]

        change_map = self.det_feat_proj(det_p2_feat)                  # [B, D, h, w]
        node_feat = F.adaptive_avg_pool2d(change_map, (G, G))        # [B, D, G, G]
        node_feat = node_feat.flatten(2).permute(0, 2, 1)            # [B, N, D]

        # ============ Step 2: 拓扑边 Transformer ============
        edge_logits, edge_scores = self.topo_net(
            node_feat, self.grid_pos, self.knn_idx
        )                                                            # [B, N, K]

        # ============ Step 3: 图路由双向精修（稀疏 KNN） ============
        enhanced = self._sparse_message_passing(
            edge_scores, node_feat, node_act_flat
        )                                                            # [B, N]

        connectivity = enhanced - node_act_flat                       # [B, N] 双向

        # 渲染回空间
        routing_grid = enhanced.view(B, 1, G, G)
        conn_grid = connectivity.view(B, 1, G, G)

        routing_map = F.interpolate(routing_grid, size=(H, W),
                                    mode="bilinear", align_corners=False)
        conn_map = F.interpolate(conn_grid, size=(H, W),
                                 mode="bilinear", align_corners=False)

        gate_input = torch.cat([routing_map, conn_map], dim=1)        # [B, 2, H, W]
        gate = torch.sigmoid(self.gate_net(gate_input))               # [B, 1, H, W]

        routing_delta = routing_map.clamp(0, 1) - p_fg                # [B, 1, H, W] 双向

        alpha = torch.sigmoid(self.alpha_raw)
        refined_p = (p_fg + alpha * gate * routing_delta).clamp(1e-6, 1 - 1e-6)
        refined_logit = torch.logit(refined_p, eps=1e-6)

        out = logit_2ch.clone()
        out[:, 1:2] = out[:, 1:2] + alpha * (refined_logit - out[:, 1:2])

        # ============ 拓扑损失 ============
        topo_loss = None
        if gt_mask is not None:
            topo_loss = self._topo_loss(edge_logits, gt_mask)

        return out, topo_loss
