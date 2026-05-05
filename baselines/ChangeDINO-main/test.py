import torch
import os
from tqdm import tqdm
from PIL import Image

from util.metric_tool import (
    ConfuseMatrixMeter,
    component_recall_scores,
    init_component_recall_stats,
    update_component_recall_stats,
)
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model

if __name__ == "__main__":
    opt = Options().parse()
    opt.phase = "test"
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data, ncols=80)
    total_iters = test_size
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()
    component_stats = init_component_recall_stats()
    eval_fg_threshold = float(getattr(opt, "eval_fg_threshold", 0.5))

    test_save_path = os.path.join(opt.checkpoint_dir, opt.name, "pred")
    if opt.save_test and not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_logits = model.inference(_data["img1"].cuda(), _data["img2"].cuda())
            # update metric
            val_target = _data["cd_label"].detach()
            val_prob = torch.softmax(val_logits.detach(), dim=1)[:, 1]
            val_pred = (val_prob >= eval_fg_threshold).long()
            _ = running_metric.update_cm(
                pr=val_pred.cpu().detach().numpy(), gt=val_target.cpu().detach().numpy()
            )
            pred_np = val_pred.cpu().detach().numpy()
            gt_np = val_target.cpu().detach().numpy()
            for pred_item, gt_item in zip(pred_np, gt_np):
                update_component_recall_stats(
                    component_stats,
                    gt_item,
                    pred_item,
                    tiny_area_thresh=int(getattr(opt, "tiny_area_thresh", 100)),
                    small_area_thresh=int(getattr(opt, "small_area_thresh", 400)),
                )
            if opt.save_test:
                for j in range(val_pred.shape[0]):
                    pred = Image.fromarray((val_pred[j].cpu().detach().numpy()*255).astype("uint8"))
                    pred.save(
                        os.path.join(test_save_path, _data["fname"][j])
                    )
        val_scores = running_metric.get_scores()
        val_scores.update(component_recall_scores(component_stats))
        message = "(phase: %s) " % (opt.phase)
        for k, v in val_scores.items():
            if k.endswith("_components"):
                message += "%s: %d " % (k, int(v))
            else:
                message += "%s: %.4f " % (k, v * 100)
        print(message)
