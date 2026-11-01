# RoadScan Dataset

Road images with bounding-box annotations for longitudinal cracks, transverse cracks, alligator cracks and potholes.

## Annotation schema

Each image has one matching text file. A row contains class ID, horizontal centre, vertical centre, width and height. Coordinates are normalized to the image dimensions. Empty files contain no annotated boxes.

## Sources

RDD2022: https://figshare.com/articles/dataset/21431547

SaudiRD2 v2: https://universe.roboflow.com/manalbitesbytes/saudird-2/dataset/2

The SaudiRD2 publisher lists CC BY 4.0. Source attribution remains applicable to redistributed data.

## Quality

The manifest and checksums identify the supplied files. Structural validation does not establish annotation completeness, geographic representativeness or independent physical scenes. Saudi annotation confirmation remains pending.
