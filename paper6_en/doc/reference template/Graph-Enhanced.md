Abstract—Polarimetric synthetic aperture radar holds significant potential for postdisaster building damage assessment. However, conventional methods often rely on simple change indices that neglect feature correlations and employ pixel-level classification models that lack spatial contextual information, leading to limited accuracy and insufficient robustness. To address these limitations, this article proposes a semisupervised building damage assessment framework that enhances performance by fusing multilevel change features with cross-graph spatial contextual information. The framework first constructs a more comprehensive multilevel change feature input that is highly sensitive to building damage. Second, a lightweight network, GESNet, centered around a graph context module, is designed. It overcomes the receptive field limitations of traditional convolutional networks by constructing graphs across adjacent image patches and propagating contextual information. Furthermore, this study establishes an iterative semisupervised learning procedure by integrating fuzzy C-means clustering with a self-training strategy to effectively mine supervisory information from unlabeled data. Experimental results on ALOS-2 and Sentinel-1 datasets demonstrate that the proposed framework significantly outperforms traditional methods, supervised baselines, and other state-of-the-art semi-supervised models across all evaluation metrics. Compared to the next-best semi-supervised model, our method achieves a 5.3% improvement in balanced accuracy.

---

RECENT years, the frequent occurrence of natural disasters such as earthquakes has posed a severe threat to global socioeconomic stability and human life [1]. After a disaster, obtaining a rapid and comprehensive assessment of building damage is crucial, as this information is often closely correlated with human casualties and directly impacts life-saving rescue operations [2], [3], [4], [5]. Synthetic aperture radar (SAR), as an active remote sensing technology, offers all-weather, all-day operational capabilities [6], making it an essential tool for post-disaster building damage assessment.

SAR-based detection of building damage from events such as earthquakes and tsunamis is primarily achieved through change detection using multitemporal SAR images acquired before and after the disaster. These techniques can be broadly categorized into single-, dual-, and fully-polarimetric methods.

Single polarimetric SAR change detection, the earliest and most widely applied method, focuses on analyzing changes in backscattering intensity. For instance, Endo et al. [7] developed a multiclassifier based on a support vector machine for tsunami damage assessment. They demonstrated that fusing multiscale features extracted from multitemporal SAR intensity images, such as the difference and correlation coefficient of backscattering coefficients, could construct a more effective high-dimensional feature space, thereby improving multiclass classification accuracy. To enhance assessment stability, Matsuoka and Nojima [8] proposed a model that integrates information on SAR backscattering coefficient changes with seismic intensity information. By combining data from these two different sources, they improved the reliability of the damage assessment. Although single-polarization methods established a foundational approach, their primary reliance on backscattering intensity or correlation makes them susceptible to noise and decorrelation effects over long temporal baselines or in complex scenes, limiting their detection reliability.

---

To acquire richer scattering information and improve the detectability of building damage, research has progressively shifted towards dual-polarization SAR change detection. Such methods move beyond simple intensity analysis to explore the amplitude and phase relationships between different polarization channels. For example, a study by Ferrentino et al. [9] jointly utilized ascending and descending orbit Sentinel-1 data. They applied two features derived from the dual-polarization covariance matrix: one based on the interchannel correlation and another based on the eigenvalues of the covariance matrix difference, demonstrating that combining ascending and descending orbit data could significantly enhance assessment accuracy. In another study, Ferrentino et al. [10] further elaborated on this novel physical approach, which involves performing an eigenvalue decomposition on the difference between the pre- and postdisaster dual-polarization covariance matrices. By using the resulting eigenvalues to quantify change, they achieved higher sensitivity to damage compared to traditional coherence-based methods. While dual-polarization information has improved detection accuracy to some extent, the incompleteness of the polarization information means there is still room for improvement in its detection capabilities.

---

Fully polarimetric SAR (PolSAR) change detection is capable of capturing the most complete scattering information from buildings. In their research on tsunami damage, Chen and Sato [11] proposed two effective polarimetric indicators: the first is the change ratio of the dominant double-bounce scattering component derived from the Yamaguchi decomposition, and the second is the standard deviation of the difference between the sorted pre- and postdisaster polarization orientation angle sequences. Building upon this, Chen et al. [12] further validated the stable linear relationship between the double-bounce scattering component ratio—a damage index—and the actual damage level, which they used to achieve quantitative damage-grade mapping. Park and Jung [13] applied fully polarimetric methods to damage caused directly by seismic ground motion. Through receiver operating characteristic analysis, they compared various polarimetric features and found that indicators representing changes in the scattering mechanism were the most effective, particularly the change in the double-bounce scattering contribution from the Yamaguchi decomposition and the change in the dominant scattering angle from the Cloude-Pottier decomposition. They also proposed a multi-indicator fusion method based on fuzzy logic. More recently, Deng et al. [14] demonstrated that a deep learning model could be used to reconstruct pseudo-fully polarimetric data from dual-polarization data. They then utilized the fluctuation features of the copolarization coherence pattern extracted from the reconstructed data for damage assessment, obtaining performance comparable to that of true fully polarimetric data. In summary, fully polarimetric methods can characterize buildings using a rich set of features, including polarimetric decomposition components and polarimetric coherence, thereby maintaining the stability and reliability of detection results even under conditions with long temporal baselines.

Despite the progress achieved in existing building damage assessment methods using PolSAR data, three critical bottlenecks remain, limiting their effectiveness in practical emergency

---

response applications. First, regarding the representation of change features, current methods commonly rely on channel-wise, independently calculated change indices such as the log-ratio [13]. Although simple to compute, such indices neglect the statistical correlations among multidimensional polarimetric feature components and thus fail to capture the holistic changes in complex scattering patterns caused by structural damage. This ultimately leads to an insufficient representation of change features, undermining the accuracy of subsequent classification models. Second, in the interpretation and classification of change information, research has predominantly focused on pixel-based traditional machine learning methods or simple thresholding [2], [7], [13]. These methods process each pixel independently, completely ignoring the spatial neighborhood relationships and contextual information between pixels. This limitation not only makes them sensitive to the inherent speckle noise in PolSAR imagery but also prevents the effective use of crucial a priori knowledge, such as the spatial clustering of damaged areas. Consequently, the damage assessment results often contain significant noise, failing to meet the high-accuracy and high-reliability requirements of emergency assessment. Finally, at the learning paradigm level, existing deep learning methods struggle to cope with the reality of extremely scarce labeled samples in postdisaster emergency scenarios. Although advanced deep learning models can address the issue of spatial context fusion [15], [16], their reliance on large-scale, pixel-level labeled data in a fully supervised mode is a major drawback [17], [18], [19], [20]. Acquiring such datasets quickly is unrealistic in the urgent context of postdisaster response. Therefore, how to effectively leverage vast amounts of unlabeled data in conjunction with extremely limited manual labels has become a core challenge restricting the practical application of advanced models.

Overall, this research is dedicated to addressing the key technical challenges in postearthquake building damage assessment using PolSAR data. To overcome issues such as the scarcity of labeled samples in emergency scenarios and the insufficient utilization of spatial information in traditional methods, this article proposes a semisupervised building damage assessment framework that fuses multilevel polarimetric features with cross-graph spatial context. The goal is to provide timely and reliable decision support for emergency rescue operations. The main innovative contributions of this article are as follows.

1) A multilevel change feature representation method is proposed. This method fuses low-level features that characterize relative change (e.g., log-ratio) with high-level features that represent statistical significance (e.g., Mahalanobis distance, Wishart distance). This approach overcomes the deficiencies of traditional single change metrics, which provide incomplete information and are sensitive to noise, thereby constructing a multichannel feature input that is more robust to damage patterns.
2) A lightweight graph-enhanced segmentation network (GESNet) centered on a graph context module (GCM) is designed. This network aims to solve the problem of limited receptive fields in deep learning models caused by image patching strategies. The GCM breaks through the visual field limitations of traditional convolutional networks by propagating contextual information across regional graphs constructed from adjacent patches, significantly improving the spatial continuity and accuracy of the segmentation results.
3) An iterative semisupervised learning framework is constructed. To address the challenge of scarce labeled samples, this framework centers on a self-training strategy. It utilizes fuzzy C-means (FCM) clustering to mine pseudo-labels from unlabeled data and periodically refines the label set through the model's own predictions. This strategy effectively mines supervisory information from unlabeled data, reduces the dependency on manual labeling, and significantly enhances the model's performance and generalization ability under few-shot conditions.

The rest of this article is organized as follows. Section II introduces our proposed method. Section III presents the experiment and results analysis. Section IV provides the discussion. Finally, Section V concludes this article.

which reflect the relative changes in features; at a high level, it integrates channels generated via Mahalanobis and Wishart distance metrics, which characterize the statistical significance of the change. Second, to effectively fuse spatial contextual information and mitigate the dependency on labeled samples, we designed an iterative semisupervised change detection framework. This process employs our proposed GESNet. The model is pretrained using high-confidence pseudo-labels obtained from unlabeled data via an initial FCM clustering. In subsequent iterations, the framework engages in self-training by continuously optimizing the pseudo-labels, leveraging the model's own enhanced prediction probability maps. Finally, the ultimate building damage assessment result is obtained by analyzing the three-class damage probability map output by the model, wherein each pixel is assigned to a "slightly damaged," "moderately damaged," or "severely damaged" category based on its class probability value. Through the construction of multilevel features, the deep fusion of cross-graph spatial information, and an efficient semisupervised learning strategy, this framework achieves highly accurate and robust building damage assessment with a limited number of labeled samples.

### A. Multilevel Difference Map Generation

1) **Polarimetric Scattering Feature Extraction**: In PolSAR imagery, the scattering characteristics of buildings are intrinsically linked to their geometric structures. Intact buildings, particularly their wall-to-ground structures that form 90° dihedral corners, produce strong double-bounce scattering, which is their primary scattering signature [21], [22], [23]. When these buildings are damaged or collapse during a disaster such as an earthquake, their regular dihedral structures are destroyed and

---

transformed into randomly oriented, fragmented rubble. This structural alteration directly changes the polarimetric scattering mechanisms. The dominant double-bounce scattering is significantly weakened, while volume scattering, caused by the randomly distributed rubble, and surface scattering, from exposed rough ground, will correspondingly increase [24], [25], [26]. Therefore, effectively tracking the changes in double-bounce, volume, and surface scattering components is central to building damage assessment. However, due to the complex nature of building damage and the partial destruction of structures, changes in a single scattering component are often insufficient to make a definitive judgment; a more comprehensive and holistic combination of features is required to analyze the changes in scattering structure.

Based on the analysis above, and to provide a comprehensive description of the building's polarimetric scattering signature, this study constructs a five-dimensional (5-D) feature vector, \( f = [P_d, P_v, P_s, F_{PPSC}, \rho_{RRLL}] \). This vector describes each image pixel and incorporates the double-bounce scattering power (\( P_d \)), volume scattering power (\( P_v \)), and surface scattering power (\( P_s \)) from the Yamaguchi decomposition [27]; the polarimetric projection-based scattering characteristics (\( F_{PPSC} \) [28]; and the circular polarization correlation coefficient (\( \rho_{RRLL} \) [29]).

---

To build upon the Yamaguchi decomposition and achieve a more refined modeling of the dihedral structure, which is critical for damage assessment, this paper introduces the \( F_{PPSC} \). This feature is more sensitive to the presence of dihedral structures than the \( P_d \) component and has the added benefit of being able to distinguish between intact buildings with different orientation angles relative to the radar's line of sight. When a building's dihedral structure is destroyed, the value of \( F_{PPSC} \) decreases significantly.

The circular polarization correlation coefficient, \( \rho_{RRLL} \), is included as a supplementary feature. This coefficient is sensitive to changes in target roughness and helicity. Building damage typically results in an increase in surface roughness and structural fragmentation, which in turn increases the magnitude of \( \rho_{RRLL} \). This feature exhibits good complementarity with double-bounce scattering features, thereby helping to improve the separability of different damage classes.

2) **Difference Map Generation**: To comprehensively capture the complex changes caused by building damage and to provide an information-rich and robust input for the deep learning model, this framework is designed with a multilevel change feature generation and fusion strategy. This strategy moves beyond the traditional approach of relying on a single change metric and

---

instead constructs and combines change features at different hierarchical levels.

First, for the pre- and postdisaster feature vectors \( f_{\text{pre}} \) and \( f_{\text{post}} \), the log-ratio operation, \( d_L = \lg(f_{\text{pre}}/f_{\text{post}}) \), is applied to represent the relative change of the features [30], [31]. This operation effectively handles the dynamic range of the feature values, focusing the change detection on relative rather than absolute changes. However, this channel-wise computation method overlooks the statistical correlation between features, making it difficult to fully characterize the combined feature changes resulting from alterations in the scattering pattern.

To solve the aforementioned problem, this article employs the Mahalanobis distance [32], [33] to comprehensively measure the change in the multidimensional feature space. It is computed through the following steps: first, the feature difference vector \( d = f_{\text{pre}} - f_{\text{post}} \) is calculated for each pixel. Then, the covariance matrix \( S \) of all difference vectors across the entire image is computed. Finally, the Mahalanobis distance \( d_M \) for each pixel is obtained using the following formula:

\[
d_M = \sqrt{(f_{\text{pre}} - f_{\text{post}})^T S^{-1} (f_{\text{pre}} - f_{\text{post}})}.
\tag{1}
\]

---

By incorporating the inverse of the covariance matrix of the differences, the Mahalanobis distance not only eliminates the influence of scale and variance among different features but, more importantly, accounts for the statistical correlation among all features. This consideration of statistical relationships shifts the focus of the measurement from independent feature magnitude changes to the combined patterns of the multidimensional feature set. Since genuine building damage induces physically meaningful, multifeature synergistic changes, whereas background fluctuations are often random and lack such patterns, the Mahalanobis distance is highly sensitive to actual damage. This allows it to effectively highlight true damage areas against a background of large-scale, statistically consistent noise.

Furthermore, this study introduces the Wishart distance [34], [35], which is widely applied in tasks such as classification and change detection. While the aforementioned Mahalanobis distance operates on multidimensional feature vectors derived from physical models, the Wishart distance operates directly on the fundamental statistical measure describing the polarimetric state—the polarimetric covariance matrix. The Wishart distance is an optimal statistical measure for assessing the similarity between two covariance matrices and is highly sensitive to changes in scattering mechanisms. It can provide a high-level change feature that is complementary to the perspective offered by the Mahalanobis distance. The formula for the Wishart distance \( d_W \) is as follows:

\[
d_W = \text{Tr} \left( C_{\text{pre}}^{-1} C_{\text{post}} + C_{\text{post}}^{-1} C_{\text{pre}} \right) - 6
\]

where \( C_{\text{pre}} \) and \( C_{\text{post}} \) are the polarimetric covariance matrices before and after the disaster, respectively.

The various change features calculated above are combined to construct a multilevel, 7-D change feature vector \( d_{\text{dif}} = [d_L, d_M, d_W] \). This 7-D vector constitutes a feature space with enhanced class separability. This characteristic serves a dual purpose: on one hand, it enables the subsequent FCM clustering

```markdown
to produce high-quality initial labels, laying a solid foundation for the semi-supervised learning framework. On the other hand, it provides the GESNet model with enhanced, multifaceted change information. This allows the model to focus on learning complex spatial context rather than extracting basic features from scratch, thereby improving the robustness and accuracy of the final assessment.

### B. Semisupervised Change Detection Framework

In Section II-A, we constructed an information-rich, multilevel change feature map. However, a direct analysis of this difference map yields suboptimal results. This is because the difference map is fundamentally a pixel-wise measure, lacking an understanding of spatial context. It cannot effectively distinguish between contiguous regions formed by genuine building damage and isolated change points caused by speckle noise, which leads to a final result that is fragmented and noisy.

To address this bottleneck while simultaneously alleviating the problem of label scarcity in emergency scenarios, this research constructs an iterative semisupervised change detection framework. The framework aims to perform an in-depth, context-aware interpretation of the multilevel change features generated in the preceding stage. Its core consists of the GESNet model and a composite loss function, which mines supervisory information from unlabeled data through a self-training paradigm to progressively optimize the model's performance. The process begins with an initialization phase that uses FCM clustering [36], [37] to generate initial pseudo-labels. Subsequently, it enters an iterative optimization phase where the model is trained under the joint supervision of a few ground-truth labels and a dynamically updated set of pseudo-labels. The model's own prediction results are periodically used to refine the pseudo-label set, thereby achieving self-evolution and performance enhancement.

---

1) **Graph-Enhanced Segmentation Network**: The GESNet proposed in this article maximizes the utilization of spatial contextual information while ensuring the overall computational efficiency of the model. The model as a whole adopts a lightweight U-Net encoder-decoder architecture. Its relatively shallow network depth and refined channel configuration ensure low parameter count and computational complexity, meeting the need for rapid response in postdisaster scenarios. Within the U-Net backbone, the key innovation of GESNet lies in the introduction of the GCM, as illustrated in Fig. 2. This mechanism is designed to break through the receptive field limitations inherent in methods based on image patches. Unlike self-attention and other mechanisms that can capture long-range dependencies but at a high computational cost, the GCM is designed to be more lightweight and efficient. It achieves effective fusion of long-range spatial information at a low computational cost by passing messages between patch nodes after a global pooling downsampling step. Its core workflow includes the following.

1) **Region Graph Construction**: A block of \( K \times K \) spatially adjacent patches, \(\{x_i\}_{i=1}^{K^2}\), is used as input. Each patch \( x_i \in \mathbb{R}^{C \times H \times W} \) is passed through the U-Net's encoder and

---

mapped to a bottleneck layer, where its features are abstracted into a high-dimensional node vector \( z_i \). Together, these nodes form a region graph \( G \) that describes the spatial relationships within this macroscopic region. Specifically, the generation process for the \( i \)-th patch's node vector \( z_i \in \mathbb{R}^D \) is as follows:

\[
b_i = \text{Encoder}(x_i)
\tag{3}
\]

\[
z_i = \text{GlobalAvgPool}(b_i)
\tag{4}
\]

where `Encoder()` represents the U-Net's encoder network, which maps the input patch \( x_i \) to a bottleneck feature map \( b_i \in \mathbb{R}^{D \times H' \times W'} \). `GlobalAvgPool()` denotes the global average pooling operation, which aggregates the feature map \( b_i \) into the final node embedding vector \( z_i \).

2) **Graph Message Passing**: A graph neural network (GNN) module is applied to the region graph \( G \). Through message passing and aggregation among the nodes, each patch is enabled to perceive and fuse the contextual information of its neighboring patches. For an arbitrary node \( i \) in the graph, its feature update process in the \( l \)-th GNN layer can be divided into two steps. First, a neighborhood aggregation is performed, gathering the feature information from its neighboring nodes \( \mathcal{N}(i) \) to generate an aggregated vector \( a_i^{(l)} \)

\[
a_i^{(l)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} z_j^{(l-1)}
\tag{5}
\]

where \( z_j^{(l-1)} \) is the input vector of the neighboring node \( j \) at the \( (l-1) \)-th layer, and \( |\mathcal{N}(i)| \) represents the number of neighboring nodes. Next, a feature update is performed. The node's own feature \( z_i^{(l-1)} \) is combined with the aggregated neighborhood feature \( a_i^{(l)} \), and a nonlinear transformation with a residual connection is applied to

---

compute the new input vector \( z_i^{(l)} \) for the \( l \)-th layer

\[
z_i^{(l)} = z_i^{(l-1)} + \sigma \left( \text{Norm} \left( W_{\text{self}} z_i^{(l-1)} + W_{\text{neigh}} a_i^{(l)} \right) \right)
\tag{6}
\]

where \( W_{\text{self}} \) and \( W_{\text{neigh}} \) are learnable linear transformation weight matrices, \( \text{Norm}(\cdot) \) represents layer normalization, and \( \sigma(\cdot) \) is the SiLU activation function. This process is repeated \( L \) times to achieve message passing across a multihop neighborhood.

3) **Context Feature Modulation**: After \( L \) rounds of graph message passing, the updated node vector \( z_i^{(L)} \) now carries rich regional contextual information. This vector is passed through a feature-wise linear modulation layer [38] to generate modulation parameters (\( \gamma_i \) and \( \beta_i \)), which are then used to perform dynamic, context-aware modulation on the corresponding patch's bottleneck feature map \( b_i \). The calculation for the modulated feature map \( b_i' \) is as follows:

\[
\gamma_i = \text{MLP}_\gamma \left( z_i^{(L)} \right)
\tag{7}
\]

\[
\beta_i = \text{MLP}_\beta \left( z_i^{(L)} \right)
\tag{8}
\]

\[
b_i' = b_i \odot (1 + s \cdot \gamma_i) + s \cdot \beta_i
\tag{9}
\]

where \( \text{MLP}_\gamma(\cdot) \) and \( \text{MLP}_\beta(\cdot) \) are two independent multilayer perceptrons used to generate the modulation parameters; \( s \) is a scaling factor that controls the modulation strength and is stabilized during training; and \( \odot \) denotes element-wise multiplication. Finally, the modulated feature map \( b_i' \) is fed into the U-Net's decoder to produce the final prediction result, which has been fused with regional context.

---

In summary, the GCM proposed in this article integrates contextual information from the local to the regional level through a closed-loop process consisting of three core steps: region graph construction, graph message passing, and context feature modulation. This mechanism combines the powerful local feature extraction capabilities of convolutional networks with the macroscopic relational modeling abilities of GNNs, enabling the model to incorporate a wider range of scene information for reference when analyzing each image patch. This "compress first, then propagate, then modulate" strategy strikes a delicate balance between performance and efficiency, providing a highly effective and feasible solution to the problem of long-range dependencies in remote sensing imagery.

2) **Loss Function**: The optimization of the GESNet model is driven by a composite semisupervised loss function, \( \mathcal{L}_{\text{total}} \), which is composed of three weighted terms. Its definition is as follows:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \lambda_{\text{hard}} \mathcal{L}_{\text{hard}} + \lambda_{\text{soft}} \mathcal{L}_{\text{soft}}
\tag{10}
\]

where \( \mathcal{L}_{\text{sup}} \) is the supervised loss, \( \mathcal{L}_{\text{hard}} \) is the hard pseudo-label loss, and \( \mathcal{L}_{\text{soft}} \) is the soft pseudo-label loss. \( \lambda_{\text{hard}} \) and \( \lambda_{\text{soft}} \) are the corresponding weighting coefficients. In this article, \( \lambda_{\text{hard}} \) is set to 0.5, and \( \lambda_{\text{soft}} \) is set to 0.1 to serve as a regularization supplement.

---

Both the supervised loss and the hard pseudo-label loss employ Focal Loss, which is intended to alleviate the class imbalance problem commonly found in change detection tasks. \( \mathcal{L}_{\text{sup}} \) is calculated on pixels with manual ground-truth labels, whereas \( \mathcal{L}_{\text{hard}} \) is calculated on the high-confidence hard pseudo-labels generated by FCM. The general form is

\[
\mathcal{L}_{\text{Focal}}(y, \hat{y}) = -\alpha_y (1 - \hat{y}_y)^\gamma \log (\hat{y}_y)
\tag{11}
\]

where \( y \in \{0, 1\} \) is the class index from the ground-truth or hard pseudo-label (0 for undamaged, 1 for damaged); \( \hat{y} = [\hat{y}_0, \hat{y}_1] \) is the two-channel Softmax probability vector output by the model; \( \hat{y}_y \) is the model's predicted probability for the correct class \( y \); \( \gamma \) is the focusing parameter, used to down-weight easily classifiable samples, forcing the model to focus on learning from difficult samples; and \( \alpha_y \) is the balancing parameter for class \( y \), used to adjust the contribution of different classes to the total loss.

To compensate for the uncertainty in hard pseudo-labels and reduce the risk of error accumulation from incorrect labeling, this study introduces the soft pseudo-label loss, \( \mathcal{L}_{\text{soft}} \). This loss term utilizes the continuous membership degree information generated by FCM to construct a mean squared error objective function with a confidence gate. This strategy not only preserves the rich information present in the membership degrees but also encourages the model to learn the uncertainty within the pseudo-labels, thereby improving the robustness of the semisupervised training process. Its definition is as follows:

\[
\mathcal{L}_{\text{soft}} = \frac{1}{2} \left( \mathbb{E} \left[ \omega_0 \cdot (\hat{y}_0 - s_0)^2 + \omega_1 \cdot (\hat{y}_1 - s_1)^2 \right] \right)
\tag{12}
\]

---

where \( s_0, s_1 \in [0, 1] \) are the membership degrees for class 0 and class 1, respectively, as calculated by the FCM algorithm. The terms \( \omega_0 = \max(0, s_0 - \tau_0) \) and \( \omega_1 = \max(0, s_1 - \tau_1) \) are confidence gate weights, where \( \tau_0 \) and \( \tau_1 \) are membership degree thresholds, both set to 0.7 in this article. \( \mathbb{E}[\cdot] \) denotes the expectation, which is the sum over all activated pixels.

3) **Semisupervised Learning Workflow**: In the task of post-disaster building damage assessment, acquiring large-scale, pixel-level labeled data is highly challenging in terms of time, manpower, and technical resources. Consequently, the volume of available manually labeled samples is often extremely limited. To address this challenge, this study adopts a semisupervised learning paradigm, which aims to effectively propagate supervisory information from a small set of labeled samples to a vast amount of unlabeled data. While ensuring model performance, this approach greatly reduces the reliance on manual labeling. The entire process follows an "initialize first, then iteratively optimize" strategy, which can be specifically divided into the following two phases.

1) **Initialization Phase**: The objective of this phase is to generate the first batch of high-quality pseudo-labels from the unlabeled data to enable the initial training of the model. The specific steps are as follows.

First, the FCM algorithm is employed to cluster the constructed 7-D change feature map. The number of clusters is set to 3, implicitly representing "unchanged," "uncertain," and "changed" patterns, with the fuzziness

---

parameter set to 2. Crucially, to establish the semantic meaning of each cluster, the generated cluster centers are sorted based on their mean intensities. The cluster with the lowest center value is defined as the "unchanged" class (background), the cluster with the highest value is defined as the "changed" class (damage). This process yields a membership matrix.

Next, a strict filtering mechanism is applied to select "high-confidence" samples for hard pseudo-label generation. We introduce a confidence threshold \( \tau \) (empirically set to 0.9). A pixel is assigned a hard pseudo-label (0 for undamaged, 1 for damaged) only if its maximum membership degree exceeds \( \tau \). Pixels with maximum membership degrees below this threshold are considered "uncertain" and are excluded from the initial training set to prevent noise accumulation. Meanwhile, soft pseudo-labels directly utilize the continuous membership degrees of the "unchanged" and "changed" classes as regression targets.

2) **Iterative Optimization Phase**: This phase constitutes the core of the semisupervised learning process. It achieves self-enhancement of the model's performance through an iterative loop of model training and pseudo-label refinement. This phase involves two alternating key steps.

**Model Training**: In each training epoch, the GESNet model is trained on a hybrid dataset composed of the few ground-truth labeled samples and the current batch of pseudo-labels. The model is optimized using the total loss function \( \mathcal{L}_{\text{total}} \).

---

**Pseudo-label Refinement**: The training is periodically paused. At this point, the improved model is used to predict building damage across the entire area. The model's output, a Softmax probability map, is generally more accurate and has a higher signal-to-noise ratio compared to the initial difference map. To ensure that the probability map is stable and incorporates multisource information, this study stacks the model's output probability map with the initial difference map at the feature level, forming a more informative, multichannel composite map. Subsequently, this composite map is used to generate higher-quality, more accurate hard and soft pseudo-labels via the same FCM process for the next round of model training. Using the FCM algorithm to reanchor the predictions based on the initial difference map serves as a form of regularization, effectively preventing the model from amplifying its own prediction errors during the self-training process and ensuring the effectiveness of the knowledge mining.

Through the aforementioned "train-refine-retrain" iterative cycle, the GESNet model progressively transfers the knowledge learned from the small set of ground-truth samples and high-confidence pseudo-labels to the broader unlabeled data. This process ultimately achieves a high-precision analysis of the entire building damage area.

---

### C. Grid-Level Assessment

To provide assessment results with greater practical value for decision-making, this study introduces a grid-level assessment metric [39]. This method first divides the study area into a regular grid of uniformly sized cells. Then, by statistically analyzing the

---

classification results of the building pixels within each grid cell \( i \), the building collapse rate (CR) for that cell is calculated

\[
\text{CR}_i = \frac{\text{PDB}_i}{\text{PDB}_i + \text{PIB}_i}
\tag{13}
\]

where \( \text{PDB}_i \) and \( \text{PIB}_i \) represent the number of pixels corresponding to damaged and intact buildings in the \( i \)th grid cell, respectively. This assessment approach, based on regional statistics, not only reflects the overall disaster severity of a region more effectively but also significantly reduces the interference of isolated misclassified pixels on the final result through a spatial smoothing effect. Finally, based on the CR value, each grid cell is classified into one of three damage grades using the following thresholds:

\[
\begin{cases} 
\text{Grid}_i \in \text{Slight damage}, & \text{if } \text{CR}_i \leq 0.25 \\ 
\text{Grid}_i \in \text{Moderate damage}, & \text{if } 0.25 < \text{CR}_i \leq 0.5 \\ 
\text{Grid}_i \in \text{Serious damage}, & \text{if } \text{CR}_i > 0.5 
\end{cases}
\tag{14}
\]

This graded quantification method corresponds to the standards used in actual on-site disaster investigations and can provide a reliable basis for decision-making in postdisaster emergency response and reconstruction planning.
```

# III. EXPERIMENT AND RESULT ANALYSIS

This section details the experimental datasets, implementation parameters, and evaluation metrics employed to assess the proposed method. Subsequently, comparative experiments and ablation studies are presented.

## A. Experimental Data

To comprehensively evaluate the performance of the proposed building damage assessment framework, this study selected two typical earthquake disaster events and conducted experiments using L-band ALOS-2/PALSAR-2 fully polarimetric SAR data and C-band Sentinel-1 dual-polarimetric SAR data, respectively. The detailed data parameters are presented in **Table I**.

The first study area selected was Mashiki town in Japan, which was severely affected by the 2016 Kumamoto earthquake. Its geographical location and the corresponding PolSAR imagery are shown in **Fig. 3**. In this case study, pre- and postdisaster fully polarimetric ALOS-2/PALSAR-2 data were used to validate the proposed method. For accuracy assessment, a reference building damage map of the area was produced based on the grid-level building damage map released by the Architectural Institute of Japan [40], which was then corrected by cross-referencing high-resolution Google Earth historical imagery from April 2016. Considering the resolution limitations of the SAR data, the building damage levels were classified into three grades: Slight (0% ≤ CR ≤ 25%), Moderate (25% < CR ≤ 50%), and Serious (CR > 50%).

The second study area is located in the disaster-stricken region of Marash, Turkey, following the 2023 earthquake, with the corresponding Sentinel-1 data shown in **Fig. 4**. This case study utilizes pre- and postdisaster Sentinel-1 dual-polarimetric SAR data as the primary data source. Acknowledging the limitations of dual-polarimetric data in describing complex scattering

---

mechanisms, we first applied the method proposed in [14] to reconstruct the original dual-polarimetric data into fully polarimetric data; the resulting pseudo-color images are shown in Fig. 4(c) and (d). The damage assessment framework proposed in this article was then applied entirely to this reconstructed fully polarimetric dataset. The reference building damage map for this case was created based on damage data released by the collaborative assessment project between Microsoft and Turkey's Ministry of Interior Disaster and Emergency Management Presidency (AFAD) [41], and was further confirmed through visual interpretation of high-resolution post-disaster Google Earth imagery from February 2023. The classification standard for the damage levels was kept consistent with that of the aforementioned Kumamoto earthquake case.

### B. Experimental Setup

In the data preprocessing stage, starting with single-look complex data, a series of steps were sequentially executed: covariance matrix generation, image coregistration, a \( 3 \times 3 \) refined Lee filter application, and geocoding. This was followed by polarimetric feature extraction using a \( 5 \times 5 \) sliding window

---

to produce the multidimensional polarimetric features for model input. The resulting feature maps were then partitioned into image patches of \( 32 \times 32 \) pixels, which served as the basic input units for the model. For the model training phase, we simulated a few-shot learning scenario. From each of the three damage categories—serious, moderate, and minor—700 labeled pixels were randomly selected as training samples, accounting for less than 3% of the total available labels. The model was trained for 30 epochs with a batch size of 64. We utilized the Adam optimizer with an initial learning rate of \( 1e-3 \) and a weight decay of \( 1e-5 \). The performance evaluation of the model was conducted at the grid level across the entire study area to assess its practical application effectiveness on a more macroscopic scale. All experiments were conducted on a workstation equipped with an AMD Ryzen 9 CPU, a GeForce RTX 4090 GPU, and 64 GB of RAM.

### C. Evaluation Metrics

To quantify the performance of the building damage assessment, we combine the imbalanced characteristics of the different damage grades and utilize class-specific metrics, including balanced accuracy (BA), Kappa coefficient, and classification

---

accuracy, recall, and F1-score for a comprehensive evaluation. The following formulas are used:

\[
\begin{cases} 
\text{BA} = \frac{1}{3} \sum_{i=1}^{3} \text{Recall}_i \\ 
\text{Kappa} = \frac{\text{OA} - \sum_{i=1}^{3} (p_{ri} \times p_{ci})}{1 - \sum_{i=1}^{3} (p_{ri} \times p_{ci})} \\ 
p_{ri} = \frac{\text{TP}_i + \sum_{j \neq i} \text{FN}_{ij}}{\text{ASN}}, \quad p_{ci} = \frac{\text{TP}_i + \sum_{j \neq i} \text{FP}_{ij}}{\text{ASN}} 
\end{cases}
\tag{15}
\]

\[
\begin{cases} 
\text{Precision}_i = \frac{\text{TP}_i}{\text{TP}_i + \sum_{j \neq i} \text{FP}_{ij}} \\ 
\text{Recall}_i = \frac{\text{TP}_i}{\text{TP}_i + \sum_{j \neq i} \text{FN}_{ij}} \\ 
\text{F1}_i = 2 \times \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i} 
\end{cases}
\tag{16}
\]

where \( i, j \in \{1, 2, 3\} \) represent the three damage classes, ASN represents the total number of samples, \( \text{TP}_i \) denotes the true positive samples of class \( i \), \( \text{FP}_{ij} \) represents the number of samples from class \( j \) misclassified as class \( i \), and \( \text{FN}_{ij} \) represents the number of samples from class \( i \) erroneously classified as class \( j \).

### D. Comparison Experiments

To comprehensively validate the effectiveness of the proposed GESNet, this study conducted both quantitative and qualitative comparisons against four representative methods in the two case study areas of Mashiki and Marash. First, to evaluate the overall advantage of our proposed framework compared to conventional technical approaches, a classic machine learning method, random forest (RF), and a standard fully supervised deep learning model (U-Net) were introduced. The inputs for both baseline models were the log-ratio change features commonly used in existing methods. Second, to ensure a fair comparison against state-of-the-art models under identical input conditions, two semi-supervised change detection models (UniMatch [42] and

---

SCGCLN [43]) were included, using the same input as our GESNet.

Table II presents the detailed accuracy assessment metrics for all methods in both study areas. In terms of overall performance, the proposed GESNet performed best among all competing methods. In the Mashiki study area, GESNet achieved a BA of 87.3% and a Kappa coefficient of 0.790. Compared to the next-best model, SCGCLN (BA = 82.0%, Kappa = 0.741), this represents an improvement of 5.3% in BA and 0.049 in the Kappa coefficient. In the Marash study area, GESNet likewise obtained the highest BA (85.9%) and Kappa (0.823). Compared to the runner-up SCGCLN model (BA = 83.2%, Kappa = 0.772), the BA and Kappa coefficient increased by 2.7% and 0.051, respectively.

GESNet demonstrated a particularly salient advantage in its ability to identify different damage levels, especially for the critical categories of moderate and serious damage. In the Mashiki region, GESNet's F1-scores for moderate and serious damage reached 74.1% and 90.9%, respectively, which are improvements of 4.5% and 5.8% over the next-best SCGCLN model. In the Marash region, GESNet's F1-scores for these two categories were also far superior, at 70.4% and 85.9%, representing increases of 2.5% and 8.2% compared to SCGCLN.

In contrast, the traditional RF method, which lacks spatial information utilization, yielded the lowest BA and Kappa coefficients in both regions. The standard U-Net model outperformed RF, but its performance was inferior to the semisupervised methods, highlighting the importance of leveraging unlabeled data in label-scarce scenarios. While UniMatch and SCGCLN, as advanced semisupervised methods, showed strong performance, our GESNet achieved further enhancements across all key metrics through its unique cross-graph context fusion mechanism.

---

The qualitative assessment results (see Figs. 5 and 6) visually corroborate the conclusions from the quantitative analysis. The grid-level results from the traditional RF method exhibit a spatially disordered distribution due to its lack of spatial awareness. The supervised U-Net shows some improvement but still suffers from a significant number of misclassifications and omissions, resulting in incomplete damage area delineation. In comparison, the proposed GESNet demonstrates spatial characteristics that are highly consistent with the reference maps in both study areas, outperforming all other methods, including the other semisupervised approaches.

### E. Ablation Experiments

To validate the effectiveness of the three core innovations proposed in this article, an ablation study was conducted, starting from a baseline U-Net model. The results of this study are presented in **Table III**.

First, the performance gain from introducing each innovation independently was analyzed. With the introduction of Innovation 1 (multilevel change features), the BA increased by 6.3% and 2.9% in the Mashiki and Marash study areas, respectively,

---

compared to the baseline model. This demonstrates that fusing low-level and high-level change information provides the model with a more discriminative input, which is fundamental to improving performance. After introducing Innovation 2 (the GCM) to create the GESNet, the BA improved by 8.2% and 1.0% in the Mashiki and Marash study areas, respectively. This indicates that the cross-patch graph interaction mechanism of GESNet can effectively aggregate spatial context, significantly enhancing the model's spatial information awareness. With the introduction of Innovation 3 (the semisupervised learning mechanism), the BA saw substantial increases of 13.4% and 4.0% in the Mashiki and Marash areas, respectively, compared to the baseline. The performance gain from this single innovation was the largest in both datasets, which strongly suggests that in scenarios with scarce labeled samples, the strategy of mining supervisory information from unlabeled data is the key to enhancing the model's generalization ability and accuracy.

Next, the cumulative effect of the innovations was analyzed. Building upon Innovation 1 by adding Innovation 2 further enhanced the model's performance. In the Mashiki region, the BA increased from 73.9% to 79.8%, a gain of 5.9%; in the Marash region, the BA rose from 78.6% to 80.7%, an increase of 2.1%. When Innovation 3 was subsequently introduced on top of Innovations 1 and 2 (representing the complete proposed method), the model's performance reached its optimum. In the Mashiki region, the BA increased from 79.8% to 87.3%, a gain of 7.5%; in the Marash region, the BA rose from 80.7% to 85.9%, an increase of 5.2%.

In summary, the ablation study clearly demonstrates that each of the three innovations proposed in this article made an independent and significant contribution to the model's performance improvement. Starting from the baseline model, each successively added innovation resulted in a step-wise increase in performance, thereby validating the rationality and effectiveness of the proposed method's design.

