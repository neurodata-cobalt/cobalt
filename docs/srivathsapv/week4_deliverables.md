# Week 4 deliverables
Srivathsa PV

## Simple Blob Detector

- Output of OpenCV's SimpleBlobDetector which is implemented based on
  - Watershed algorithm
  - OTSU's binarization threshold
  - Scale Invariant Feature Transform (SIFT)
- False positives are very less
- Can get very good results if combined with [3D reconstruction algorithms](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3873529/)

[Python Notebook](https://github.com/NeuroDataDesign/clarity-f17s18/blob/master/source/srivathsapv/opencv-blob/Simple%20Blob%20Detector.ipynb)

![](https://user-images.githubusercontent.com/1017519/30785930-36594b36-a13c-11e7-816f-5fb04fd8a870.png)

## Distance transform for overlapping cells

- Morphological opening is used to create sure background area and distance transform is performed for sure foreground area.
- This removes ambiguity between overlapping cells and identifies them individually

[Python Notebook](https://github.com/NeuroDataDesign/clarity-f17s18/blob/master/source/srivathsapv/blob-overlap/Cell%20Overlap.ipynb)

![](https://user-images.githubusercontent.com/1017519/30785927-2dc4b5dc-a13c-11e7-9e38-8f615be81c4d.png)

## Next week

- Based on the sprint timeline discussion, pick one of the unsupervised algorithms and start implementing it
- Validate the algorithm’s performance on the validation sub-volume that is created
