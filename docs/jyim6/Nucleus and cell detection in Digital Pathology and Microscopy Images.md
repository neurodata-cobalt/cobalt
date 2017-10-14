# Robust Nucleus/Cell Detection in Digital Pathology and Microscopy Images
A review of the different state of the art techniques for Nucleus/cell detection in Digital pathology and Microscopy Images. The paper is [here](https://www.ncbi.nlm.nih.gov/pubmed/26742143). The review includes segmenation as well but segmentation is not the goal of this sprint so we leave it out for now.

There are many difficulties in robust cell detection such as 
* Noise
* Artificats 
* Poor contrast between foreground and background
* Significant variation on cell shapes, sizes, intensities
* Clustering/overlap of cells

While the techniques described aim to overcome each of those challenges, there is no consensus on which methodology is better. This is because of the lack of a common evaluation datasets for each of the mentioned techniques. While one algorithm may perform exceptionally well in one setting, there is no guarantee it will achieve the same performance on a different type of image/data.

Seminal papers of each technique can be found in the review paper.

### Preprocessing 

Preprocessing is briefly mentioned but beyond the scope of the review. The preprocessing techniques of many of the algorithms are:
* Color normalization
* Image denoising
* Extraction of regions of interest

# Cell and nucleus detection methods
## Distance transform (DT)
A distance transform assigns each pixel/voxel with the distance to the nearest feature point (usually a edge point). The euclidean distance is usually chosen as the metric which results in a euclidean distance transform (EDT). The local maximas in a distance transform map usually correspond to centroids of nuclei or cells. EDT is often paired with watershed segmentation since the inverse distance map eliminates bad local minimas. 

#### Pros
* Very easy to implement
* Can be used in conjunction with other algorithms such as watershed
* Achieves fairly good results with simple images
* Effective only at identifying regular shapes in the image 
  * Maybe if we try changing up the distance metric?

#### Cons
* Sensitive to variation in the feature points (edge pixels)
* Can perform poorly at detecting overlapping cells

Therefore while DT is effective at performing simple detection and segmentation, it is not suitable for harder, more complex cells present in images like hisopathological images.

#### Paper links
[A hybrid 3D watershed algorithm incorporating gradient cues and object models for automatic segmentation of nuclei in confocal image stacks (2003)](https://www.ncbi.nlm.nih.gov/pubmed/14566936)

## Morphology operations
A morphological filter is a image processing technique that does combinations of different filtering operations such as erosion, dilation, opening, and closing to uncover some geometric or topological structure of objects in images. Some popular morphological operations are top-hat, bottom-hat, and ultimate erosion (UE) transforms. 

#### UE
UE is popular for marker detection. It repeatedly applies a series of erosion operators to each connected coponent until one more erosion will completely remove the component (not necessary remove, but separate them). This results in touching or overlapping objects to be separated. (However isn't this no different than distance transform?) The downside of the technique is the technique might overlabel objects with more than 1 marker in noisy images. 

#### Conditional erosion method
A variation of UE is using a set of multiple filters: four different 7 x 7 mask structure for corase erosion and two 3 x 3 filters for fine erosion. The coarse erosion performs erosion while maintaining object shape and fine erosion avoid undersegmentation. The method uses two thresholds to decide when to stop doing the repeated erosions with the filters. Hence the difficulty with this method is finding the right parameters for the image modality. 


#### Pros
* Fast and easy to implement (just need to run a bunch of filters)
* Performs relatively well on certain image sets when the parameters are known and binarization is known to be good.

#### Cons
* Relies on grey-scale binarization of the images. Perfect Binarization is non-trivial in some images.
* Sensitive to noise and can overdetect in noisy images
* Requires parameter finding

#### Paper links
[Automated detection of cell nuclei in pap smear images using morphological reconstruction and clustering (2011)](https://www.ncbi.nlm.nih.gov/pubmed/20952343)

[Towards automated cellular image segmentation for RNAi genome-wide screening (2005)](https://www.ncbi.nlm.nih.gov/pubmed/16685930)

## HIT/HAT
Closely related to Mophological operations, the HIT operation adds a depth value *h* and then performs an erosion operation that suppresses all the regional minimas whose depth is not larger than *h*. HAT is similar except it supresses the regional maxima whose height is not larger than *h*. The result of HIT/HAT is similar to DT except it removes local minimas caused by uneven object shapes or noise and generates more correct markers. Like DT this technique is used in conjunction with other algorithms like watershed.

HIT/HAT is known to be more stable than DT and is used more widely. The issue is in finding the threshold *h*. Certain parameter finding methods were proposed and can be found in the review.

#### Pros
* Similar to DT and Mortphological operations, can be fast and easy to implement
* Algorithms to automatically determine the *h* parameter exist
* Can be used in conjunction with algoritms like watershed for segmentation
* Popular

#### Cons
* Requires finding the parameter *h* which is non trivial even with the parameter finding algorithms.

#### Paper links
[Segmenting clustered nuclei using H-minima transform-based marker extraction and contour parameterization. (2010)](https://www.ncbi.nlm.nih.gov/pubmed/20656653)

[A Method for Automatic Segmentation of Nuclei in Phase-Contrast Images Based on Intensity, Convexity and Texture (2014)](http://ieeexplore.ieee.org/document/6762958/)

## LoG Filtering
One of the most popular methods to identify small blob objects in images. The method performs a convolution of the original image with a laplacian operator and a gaussian kernel with a scale parameter to handle object scale invariance. One popular technique is to do a multi-scale LoG blob detector at multiple scales. The most popular methods involve some sort of combination of LoG with another technique to determine the best parameters to use. The most recent methods involve using variations of the Laplacian filter itself such as generalized LoG or Hessian based convolution filters. 

#### Pros
* Fast to perform (just a bunch of convolutions)
* Very popular for blob detection and therefore cell detection
* Can be used in conjunction with a bunch of other methods.

#### Cons
* Requires parameter finding (filter scale, gaussian parameters, normalizing constant)
* In almost all applications it has to be combined with another technique for more refinement

## MSER Detection
Like LoG, Maximally stable extremal regions (MSER) is used primarily in blob detection. The idea is to generate a set of nested extremal regions based on the level sets in the intensity landscape of an image and determines if these regions are cells -- the cell regions are typically the maximally stable regions. The technique seems math/numerically heavy as in the method in determining if a region is a cell can be done in many ways such solving an optimization problem, using a statistical model to classify regions. Furthermore the region selection has to be done in careful ways to minimizing overlap and be dense enough to get all the cells. 

#### Pros
* Gives a flexible framework for detecting blobs
* Results from some papers claim the method does well in settings where cells are tightly overlapping

#### Cons
* Requires empirically finding the parameters to determine if regions are cells
* Often requires solving an optimization problem

## Hough Transform
Cells and nuclei exhibit large variation in shape but are usually in some sort of elliptical or circular shape. The hough transform makes this assumption and attempts to find the parameters that match the shape of the cell. It computes a hough-transformed image of cell candidates and runs a voting scheme for the different parameters of the shape. The technique then finds the peaks in the transformed parameter space and then determines if the candidate is a cell. The technique is traditionally known to be applied for canonical shapes like ellipsoids but can be extended to any arbitrary shape.

#### Pros
* Can be applied to any arbitrary cell shapes. Doesn't have to assume the shapes are elliptical. 
* Good for reliant blob detection
* 3-D generalied HT is already developed

#### Cons
* Can be slow
* Seems other methods always couple HT with something else
* Might generate false peaks due to image noise, incorrect edge extraction, or touching objects. 

## Radial-symmetry based voting (RST)
Another technique to locate centroids of cells. However it is computationally intensive but a Fast RST was developed to do fast detection. The idea of the technique is to map the input image into a transformed image where points with high radial symmetry are highlighted. The points with higher responses are marked as centroids. 

Honestly I don't know why this is used instead of HT. The method doesn't work well for non-circular images. All the techniques seem pretty old as well. Not going to bother writing the pros and cons. The review listed almost no pros.


## Supervised learning
Techniques in Machine learning incolve inferring a mapping function from training data. In cell detection we wish to classify cells from pixel/voxel information. The techniques described in this paper are SVM, Random Forest, and Deep neural networks. Since our interest is in unsupervised methods, I won't expand on these methods for now.




