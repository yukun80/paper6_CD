Ref.: "HA-CQI and CFDepth for Urban Flood Extent Mapping and Change-Defined Depth Estimation from Bi-Temporal SAR Imagery" (Professor Hong Zhang)


Dear Professor Zhang,

I very much regret to have to tell you that publication entitled, "HA-CQI and CFDepth for Urban Flood Extent Mapping and Change-Defined Depth Estimation from Bi-Temporal SAR Imagery" (Professor Hong Zhang) in our journal is not recommended. An explanation for this decision is given in the attached review reports (and on
https://www.editorialmanager.com/hydrol/). I hope that the comments contained therein will be of use to you.

Thank you for your interest in our journal.

With kind regards,

Dan Lu
Editor
Journal of Hydrology
...........................................................
Important note: If a reviewer has provided a review or other materials as attachments, those items will not be in this letter. Please ensure therefore that you log on to the journal site and check if any attachments have been provided.

COMMENTS FROM EDITORS AND REVIEWERS



AE: This manuscript has been evaluated by two reviewers. Based on their comments and my own assessment, I do not recommend publication for the following reasons:
1) The manuscript is presented primarily as a computer vision study rather than a hydrology-focused contribution. The methodological novelty is not clearly articulated, and Sections 3 and 4 contain excessive algorithmic details that obscure the main scientific contribution and make the manuscript difficult for a hydrology audience to follow.

2) The manuscript does not adequately explain why the proposed method outperforms the baseline models. Although superior performance is reported, the underlying reasons for these improvements are not sufficiently analyzed or interpreted. A more insightful analysis and discussion of the model's performance are needed.

3) The reviewers have also raised several additional concerns, which are detailed below.


Reviewer #1: Comments on HYDROL81461:

The manuscript is essentially a combination and application of existing methods. While the research holds some merit, the authors have failed to clearly articulate the novelty of their work. Additionally, there are numerous issues regarding the writing and presentation. Therefore, I do not recommend publication in its current form, but would encourage the authors to resubmit after a thorough revision.

Specific Comments:
1. What is the specific purpose of the "Related Work" in Section 2? It reads like a general literature review and appears to overlap with parts of the Introduction.
2. Section 3 currently details a large number of methods, most of which are already mature and well-established. I suggest simply citing these established methods rather than describing them in detail. This section should focus primarily on highlighting the methodological innovations of this work and how they are implemented.
3. Much of the content in Section 4 actually pertains to the methodology. The authors need to reorganize the structure of the paper accordingly.
4. The majority of Section 4 consists of comparisons of computational results. While these are fundamental, it is more critical for the authors to clearly demonstrate comparisons specifically targeted at the problem to be solved. This aspect is currently insufficient.
5. The "Discussion" section is too brief and lacks an in-depth analysis of the results.



Reviewer #2: The manuscript presents an interesting and timely contribution to urban flood mapping. The authors propose a two-phase flood modelling framework based on bi-temporal SAR imagery. In the first phase, newly inundated flood extent is detected using the HA-CQI module. In the second phase, flood depth is estimated from DEM-based constraints using the CFDepth module. The general structure of the paper, the presentation of the results, and the discussion are promising. However, I still have several concerns, mainly regarding the robustness, physical validity, and validation of the proposed flood-depth estimation method. In my view, the manuscript requires further clarification and stronger validation before it can be considered reliable.

Major comments

My main concern is related to the flood-depth estimation component. The CFDepth module depends directly on the flood mask produced in the first phase. Therefore, any error in flood extent detection will be transferred directly into the depth estimation. For example, if a dry but dark road surface is falsely classified as flooded in the SAR-based flood mask, the CFDepth module will still assign a water depth to that location. This means that false positives in the flood extent map may produce physically unrealistic depth values. The authors should explicitly discuss this error propagation and, if possible, quantify how sensitive the estimated depth is to errors in the flood mask.

The flood-depth estimation relies on several strong assumptions that may not always hold in urban flood environments. These include the assumptions that the water surface is smooth, that flood boundary elevation can represent the local water level, that the DEM is sufficiently accurate, that flooded pixels are hydraulically connected, that there are no strong local flow gradients, and that underground drainage, pumping systems, culverts, or other urban hydraulic structures do not significantly affect the water level. These assumptions may be acceptable in some slow-moving floodplain conditions, but they are more problematic in complex urban areas. The authors should clearly state the limitations of these assumptions and explain under what flood conditions the method is expected to perform well or fail.

The assumption of a smooth or nearly flat water surface is particularly important. Urban floods are often affected by streets, buildings, embankments, local depressions, drainage channels, blocked culverts, and stormwater systems. Two streets may both appear flooded in SAR imagery, but one may be hydraulically connected to a drainage channel while the other may be an isolated ponded area. In such cases, assigning or interpolating a similar water level may be incorrect. The manuscript should include a more careful discussion of hydraulic connectivity and local discontinuities in urban flood depth estimation.

The temporal interpretation of the estimated flood depth also needs clarification. SAR imagery captures the flood condition at one specific acquisition time, whereas the DEM is static and the flood process is dynamic. If the SAR image was acquired after the flood peak, the estimated depth represents the water depth at the image acquisition time, not the maximum flood depth during the event. This distinction is very important for flood damage assessment and emergency response. The authors should clearly state that the estimated depth is time-specific and should not be interpreted as peak flood depth unless additional evidence is provided.

lines 649-726: The validation of CFDepth is still not sufficiently convincing. The authors mainly use valid depth ratio, WSE-gradient ECDF, and qualitative field-photo intervals. These indicators show that CFDepth can produce more complete and smoother depth maps, but they do not prove that the estimated depths are numerically accurate. The photo-based comparison on page 14, lines 662-682, is useful, but the depth intervals appear to be visually inferred from objects such as cars, people, and traffic signs. This is helpful as qualitative evidence, but it is not equivalent to validation using independent water-level gauges, surveyed flood marks, UAV/LiDAR data, or in-situ depth measurements. The authors should avoid overstating the accuracy of the depth results unless stronger independent validation is provided.

The valid depth ratio and WSE-gradient ECDF should not be presented as direct evidence of depth accuracy. The valid depth ratio only shows how many flood pixels receive a positive finite depth value. It does not indicate whether those values are correct. A method can increase valid depth coverage by enforcing positive depths, but this may create plausible-looking yet inaccurate results. Similarly, the WSE-gradient ECDF mainly reflects local smoothness of the estimated water surface. A smoother water surface is not necessarily more accurate, especially in urban areas where drainage infrastructure, roads, embankments, buildings, and local barriers may create real discontinuities. The authors should revise the interpretation of these metrics and clearly distinguish completeness, smoothness, and physical accuracy.

The training and evaluation procedure is not sufficiently described for reproducibility. The authors state that VarFloods and S1GFloods were integrated, but this description is too general. The manuscript should explain how many samples were used, how the training and validation split was designed, whether tiles from the same flood event appeared in both training and validation sets, how labels from different datasets were harmonized, how SAR intensity values were normalized, and whether data augmentation was applied. Without these details, it is difficult to judge the robustness and generalizability of the proposed method.

The flood labels used for the two GF3 evaluation scenes need to be explained more clearly. The authors show flood labels, but it is not clear how these labels were produced. Were they manually interpreted from SAR imagery, derived from optical imagery, taken from official flood products, based on field reports, or generated through multi-source interpretation? Were permanent water bodies removed before evaluation? How was uncertainty in the reference labels handled? These details are essential because the reported accuracy depends strongly on the quality of the reference flood maps.
Some values in Table 2 appear unusually regular and should be carefully checked. For several baseline methods, Precision is exactly or nearly four percentage points higher than F1, and some results show a suspiciously patterned structure. This may be correct, but the authors should verify the calculations and ensure that there are no reporting, rounding, or copying errors in the table.

The semantic calibration step using DINOv3 is under-explained. Since DINOv3 is generally designed for image feature extraction, the manuscript should explain how SAR imagery is converted into a suitable input format. Are SAR channels repeated? Are intensity values normalized or transformed before being passed to DINOv3? Are the DINOv3 features pre-trained on natural images, remote sensing images, or SAR-specific data? The authors should clarify this part because the transfer of semantic features from optical or natural-image domains to SAR flood mapping is not straightforward.

The photo-based depth intervals should include uncertainty bounds and a short explanation of how object heights were estimated. For example, if cars, people, traffic signs, walls, or road structures were used as visual references, the authors should explain the assumed object heights and the possible uncertainty range. This would make the qualitative validation more transparent and more defensible.



Minor comments

The authors should revise some claims related to flood-depth accuracy. At present, the results mainly demonstrate consistency, completeness, and smoothness, but not necessarily independent accuracy. The wording should be adjusted accordingly.

The limitations of the method should be expanded, especially regarding DEM quality, SAR acquisition timing, urban drainage systems, disconnected flooded areas, false flood detections, and uncertainty propagation from flood extent to flood depth.

Chen et al. 2022a/2022b and Daudt et al. 2018a/2018b appear to be duplicated with the same title and DOI. Please check the reference list carefully and remove or correct duplicate entries.



At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

#AU_HYDROL#

To ensure this email reaches the intended recipient, please do not delete the above code