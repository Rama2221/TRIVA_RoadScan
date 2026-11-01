# RoadScan

Road damage detection and repair priority assessment for maintenance review.

| Stage | Purpose |
| --- | --- |
| Data preparation | Validate images, class mappings and bounding-box annotations. |
| Classification | Identify image-level damage categories with a ResNet18 baseline. |
| Detection | Locate damage with the fine-tuned RF-DETR Small model. |
| Priority | Rank reports using damage type, relative box size and defect count. |
| Similarity | Retrieve candidate related reports using frozen DINOv2 features and location metadata. |
| Application | Present detections, priority and maintenance report review. |

The detector checkpoint and recorded validation selection are included. Quantitative findings are stored in the results directory. RDD2022 and SaudiRD2 source attribution is retained in the dataset package.

Application source: `notebooks/application/app.py`.
