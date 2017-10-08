# Outline of Yousef's Automatic nuclei detection and segmentation algorithm
## Background
From its website: *FARSIGHT is a collection of Quantitative Tools for Studying Complex and Dynamic Biological Microenvironments from 4D/5D Microscopy Data.*
The toolkit is maintained and contributed to by different laboratories across multiple universities. The Principal investigator is Badri Roysam, department chair of the Computer Engineering department at the Univesity of Houston.

The tool in FARSIGHT relevant to our team is its Image Segmentation Library, specifically the [nuclear segmentation algorithm](http://www.farsight-toolkit.org/wiki/Nuclear_Segmentation), [paper link](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5306149). 

We became interested in this algorithm from a [survey](https://www.frontiersin.org/articles/10.3389/fnana.2014.00027/full) on unsupervised cell counting algorithms which concluded FARSIGHT is one of the best toolkits and algorithms out there. However this was back in 2014 but a more up to date survey wasn't found. The algoritm also is cited 400 times making it one of the more popular cell segmentation algorithms.

## Pipeline
![](https://user-images.githubusercontent.com/8682187/31311050-1e125492-ab72-11e7-866c-c4a77eba455c.png)

For our purposes only focus on the first box (up to seed detection). Also the paper covers the 2D methodology and that will be discussed here. The authors extended their pipeline to 3D in the FARSIGHT implementation. A paper or explanation of the extension was unable to be found.

### Raw data
The study used DAPI stained histopathology slides. Histopathology images have quite different properties than our CLARITY images but share the same blob like properties that make investigating this algorithm worthwhile. 

### Automatic initial binarization with graph cuts
First the Image is thresholded by separating the foreground from background cells. In a nutshell, this is important to know which pixels are candidates to be cells or not (e.g. can be noise or artifacts). 

Binarization (thresholding) is first done using the Poisson-distribution-based minimum error thresholding algorithm. The technique is found [here](http://www.worldscientific.com/doi/abs/10.1142/S0218001491000260) with a more in-depth mathematical derivation [here](http://www.sciencedirect.com/science/article/pii/S0167865598000282) (the math in the original paper was very low-effort) (Also if github markdown supported latex then I would tex it up here). After estimating the parameters, the poisson mixture model gives us a probability density function of whether a pixel is in the foreground or background. 

The poisson model provides a model for the binarization. Now the actual blobs are discovered using a graph cut max-flow/min-cut algorithm based on a labelling energy function. The energy function is designed to minimize the cost of giving a label to a pixel based on the poisson model while also having a continuity constraint to incorporate the assumption of convexity and connectivity in blobs. They use the popular graph-cut algorithm by [Boykov and Kolmogorov](http://www.csd.uwo.ca/~yuri/Papers/pami04.pdf). The result of this gives us a set of connected components, (e.g. a labelling of each component a cell is connected to).

#### Remark 
Normally this would be the end of blob detection. But histopathology images have the charactertics of intrinsic blobs inside larger blobs. Many cells are squished together in large clusters. The next step attempts to find these "seeds" inside the blobs found from the graph-cut. While we don't know if CLARITY data exhibits this similar phenomenom, it's worth investigating the behavior of this next step on our CLARITY data.

### Automatic seed detection and initial segmentation
Within the blobs from the previous step, they run a multiscale Laplacian of Gaussian (LoG) filter suggested by (Lindeberg)[http://ftp.nada.kth.se/CVAP/reports/cvap198.pdf]. The algorithm involves testing each pixel against LoG of various scales. A response map is calculated as the scale the LoG got the maximum response from. A euclidean distance map contraints the maximum scale value at each point. This alleviates the issue of false positives when 2 or more nucleis can get labelled as one since the distance map constrains those two nuclei from being compared on the same scale. For histography images, the filter response map has local peaks wherever nucleis are. Therefore by finding the highest response in local neighborhoods, the "seed" points or the centers of those nuclei can be detected. 

An initial segmentation of the whole nucleis are done with a size-constrained clustering algorithm described by [Wu](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.565.8960&rep=rep1&type=pdf). Watershed can be used as well but the downside is the algorithm's sensitivity to small peaks in the response map. However, the clustering algorithm requires a resolution parameter for the search radius of clusters. The best rsolution parameter has to be found empirically so that it overcomes the problem of watershed in joining together multiple nuclei. 

## Basic pseudo code
1. Fit the data to a bimodal poisson mixture model, i.e. find the threshold parameter. Get the bi-modal poisson PDF for whether a cell is in the foreground or not
2. Run a max-flow/min-cut algorithm to discover the connected components (i.e. large blobs)
3. Run the multiscale LoG for different scales and construct a response map. 
4. Find the local maximas in the response map. 
5. Run watershed or some clustering algorithm to do an initial cell segmentation. 

Note we don't cover the last steps which involve refining the segmentation and graph coloring. These aren't necessary for CLARITY (afaik).



