# FARSIGHT nuclei cell segmentation Benchmarks
# Background
From its website: *FARSIGHT is a collection of Quantitative Tools for Studying Complex and Dynamic Biological Microenvironments from 4D/5D Microscopy Data.*
The toolkit is maintained and contributed to by different laboratories across multiple universities. The Principal investigator is Badri Roysam, department chair of the Computer Engineering department at the Univesity of Houston.

The tool in FARSIGHT relevant to our team is its Image Segmentation Library, specifically the [nuclear segmentation algorithm](http://www.farsight-toolkit.org/wiki/Nuclear_Segmentation), [paper link](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5306149). 

We became interested in this algorithm from a [survey](https://www.frontiersin.org/articles/10.3389/fnana.2014.00027/full) on unsupervised cell counting algorithms which concluded FARSIGHT is one of the best toolkits and algorithms out there. However this was back in 2014 but a more up to date survey wasn't found. The algoritm also is cited 400 times making it one of the more popular cell segmentation algorithms.

# Pragmatics 

The toolkit is available for [download](http://www.farsight-toolkit.org/wiki/Special:FarsightDownloads) and the [source code](https://github.com/hocheung20/farsight) is open source. However the last commit was 4 years ago so it seems to be abandoned. 

Installation is easy, at least on OSX. Run the package and then add the binaries to your path. The only dependency needed is python. 

### Environment
The actual algorithms are implmeented in C++ but are glued together by a scripting language which they chose as Python (hurray!). This means we can utilize their code in our pipeline. 


Unfortunately the algorithm requires many parameters but provides a usage where the parameters are estimated for you. Descriptions of each parameter can be found in the paper and are outputted when the algorithm is ran which will be shown below.

Since FARSIGHT is a toolkit, a user can script together the different tools and modules and run them however they want -- GPUs, multithreads, parallel, etc. There are options to control how much memory usage to take as well.

### Documentation 
The website linked above has documentation about FARSIGHT and the paper link gives descriptions about each parameter and the algorithm pipeline. 

# Benchmarks

## Datasets
| Dataset number | Dataset name | cell count | noisy | center distribution | cell intensity | type of data
| ------- |:--------------:|:------------:|:---------------:|:-----------:|:-------------:|:---------:|
| 1       | solid_45_cells                              | 45    | No        | fixed array   | solid         | generated
| 2       | solid_45_cells_noise_random_intensity       | 45    | Yes       | fixed array   | solid         | generated
| 3       | solid_147_cells_img_noise                   | 147   | Yes       | fixed array   | solid         | generated
| 4       | blurred_147_cells                           | 147   | No        | fixed array   | gauss blurred | generated
| 5       | blurred_147_randomized_gauss_cells          | 147   | No        | gaussian      | gauss blurred | generated
| 6       | faded_147_randomized_cells_random_intensity | 147   | No        | uniform       | faded         | generated
| 7       | faded_147_randomized_cells                  | 147   | No        | uniform       | faded         | generated
| 8       | faded_147_randomized_gauss_cells            | 147   | No        | gaussian      | faded         | generated
| 9       | s3617_cutout                                | ??    | artifacts | random        | ??            | generated


<!-- * Toy datasets (listed in order from easiest to hardest):
    1. solid_45_cells.tif
    2. solid_45_cells_noise_random_intensity.tif
    3. solid_147_cells_img_noise.tif 
    4. blurred_147_cells.tif
    5. blurred_147_randomized_gauss_cells.tif
    6. faded_147_randomized_cells_random_intensity.tif
    7. faded_147_randomized_cells.tif
    8. faded_147_randomized_gauss_cells.tif

* Real (test) datasets:

    9. s3617_cutout.tif -->

Each of these images are in uint8 and are 1000 x 1000 x 100 (X by Y by Z).
Unfortunately we didn't have manually labelled datasets to use as ground truth

### Characterizations of each dataset


Command ran:
```
./segment_nuclei <input_image_path> <segmented_image_path>
```

## Sample output:

```
reading input image...done
Clearing binarization stuff
Allocating 190.735 MB of memory for binImagePtr
Start Binarization ...
Entering MinErrorThresholding
Getting im Iterator
Copying image into itkImage
maxv = 255 minv = 0
Running MinErrorThresholdingFilter
mem_size: 72 GB
image_size: 100000000
mem_needed: 47.6837 GB
num_blocks_preferred: 16
block_divisor: 2.51984
num_blocks_C: 3 num_blocks_R: 3 num_blocks_Z: 3 cntr: 27
num_pixels_per_block_R: 396 num_pixels_per_block_C: 396 num_pixels_per_block_Z: 39
Total Blocks: 27
Image size: 1000x1000x100
Starting Graph Cuts
Cell Binarization refinement by alpha expansion took 24.3621 seconds
Entering getConnCompImage
Copying input image into ITKImage
Computing the labeled connected component image
Setting minimum object size
Writing output of relabel coloring filter to input image
Returning number of connected components
Entering getConnCompInfo3D
Cell Binarized.. with 45 connected components
Computing distance transform...done
About to enter Estimating parameters
Estimating parameters...
Estimating parameters took 0.061382 seconds
    Minimum scale = 1
    Maximum scale = 2
    Clustering Resolution = 3
LoG block 1 of 1
Processing scale 1
Scale 1 done
Processing scale 2
Scale 2 done
Multiscale Log took 84.1975 seconds
Detecting Seeds
Local maxima point detection took 27.0908 seconds
done detecting seeds
3600 seeds were detected
Starting Initial Clustering
scale_xy = 3
scale_z = 2
max_nghbr_im initialized
Max_response array done
Initial max_nghbr_im took 8.24562 seconds
Entering main Clustering Loop
change=13967703
change=10139517
change=7367031
change=3798405
change=174780
change=0
Save the image to /Users/Jason/Developer/Classes/NeuroData/clarity-f17s18/source/jyim6/img/solid_45_cells_seg_final.dat in the IDL format ...
Time elapsed is: 187.703 seconds
```

To concisely summarize this algorithm, we'll only report some of the outputs. The results of running the nuclear segmentation algorithm with the system automatically calculating the parameters is given below:

| Dataset | Detected connected components | Actual cell count | Detected seeds  | Total time (seconds) | Multiscale log time (seconds) |  Local maxima point detection time (seconds) |
| ------- |:--------------:|:------------:|:---------------:| :-------------------:|:-------------------:|:-----------:|
| 1       | 45           | 45              | 3600            | 187                  | 84       | 27
| 2       | 37           | 45              | 2290            | 187                  | 81       | 27
| 3       | 147          | 147             | 237             | 632                  | 154      | 193
| 4       | 49           | 147             | 264             | 644                  | 149      | 199
| 5       | 175          | 147             | 311             | 474                  | 159      | 85
| 6       | 129          | 147            | 130             | 441                  | 158      | 85 
| 7       | 43           | 147            | 45              | 749                  | 156      | 244 
| 8       | 96          | 147            | 148             | 424                  | 84       | 158 
| 9       | 405          | 147            | 977             | 330                  | 27       | 27


## Selected results
NOTE: All the dots are seed centers. Ignore the different colors.

### **solid_45_cells_noise_random_intensity**


 Front             |   Side
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/8682187/31305857-553da7ca-ab11-11e7-9019-b8f2c14c1d1f.png) |![](https://user-images.githubusercontent.com/8682187/31305858-553e911c-ab11-11e7-856c-12d2e47b8bf6.png)

* **Precision: 93%**
* **Recall: 1.9%**

#### Discussion
* The detected connected components are exactly the blobs in this picture except the algorithm misses 8 blobs (only gets 37/45). The ones it missed are too light and might've been hampered by the salt-pepper noise that interferred with the thresholding (more investigation would be needed to confirm that). Therefore the precision is quite high. The connected component part of the pipeline does well in finding the cells.
* However the recall is very low. This is because of the explosion of detected seeds. Since the cells have homogenous intensity, the characterics of histopathology cells doesn't apply here. Histopathology cells assume heterogenous responses within the clusters. Therefore each pixel in the connected components were classified as a cell. Maybe if we do strict local maximas that would fix the seed explosion.



### **faded_147_randomized_cells_random_intensity**

 Front             |   Side
:-------------------------:|:-------------------------:
 ![](https://user-images.githubusercontent.com/8682187/31305861-553f0520-ab11-11e7-9c7b-b2612d0b6392.png)|![](https://user-images.githubusercontent.com/8682187/31305859-553ebaca-ab11-11e7-9fdb-59dc361d17d1.png)

* **Precision: 88%**
* **Recall: 100%**

#### Discussion
* The cells in this image are much more dispersed than the previosu image and smaller. In this scenario the cells are also heterogenous as in the intensity fades as you move away form the centers. Therefore the seed detection algorithm works in finding those local maximas and we don't see a explosion of seeds. That's why we see a recall of 100%. All the found centers were spot on.
* However it was unable to find all of the cells. This may be attributed again to the lack of heterogenous structure in overlapping cells. Histopathology images are stained and clear edges are seen but here the clusters are not clearly separated:

 Histopathology             |   Generated
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/8682187/31311432-18c01968-ab7a-11e7-860a-be08d0a2c236.png)|![](https://user-images.githubusercontent.com/8682187/31311433-18c1c57e-ab7a-11e7-8b73-97597d549de7.png)

To improve this we mght need a better parameter search than what the algorithm does automatically.

### **faded_147_randomized_gauss_cells**

 Front             |   Side
:-------------------------:|:-------------------------:|
 ![](https://user-images.githubusercontent.com/8682187/31306018-feb66f46-ab14-11e7-92b3-e28d00e6f2b8.png)|![](https://user-images.githubusercontent.com/8682187/31306017-feb2078a-ab14-11e7-9eab-455902226bb2.png)

* **Precision: 93%**
* **Recall: 100%**

#### Discussion
* The result of this experiment is similar as above but much more overlap. It does very well however, still having the problem of missing some overlapping cells.


### **s3617_cutout**

 Front             |   Side
:-------------------------:|:-------------------------:
 ![](https://user-images.githubusercontent.com/8682187/31305862-553f203c-ab11-11e7-9222-5e10d491dc37.png)|![](https://user-images.githubusercontent.com/8682187/31305860-553ec2fe-ab11-11e7-9075-ba5fcf45957a.png)

**Precision and recall not available.**

#### Discussion
* The result of running on actual data. We do not have ground truth numbers to see how well the algorithm does. 
* One thing to note is the structure of the cells. They have large homogenous centers but fading away from the centers so they match a larger gaussian distribution with small variances. We should try to generate a dataset that follows that phenomena closely and see how well it does.
* It will be interesting to see how well the connected components number does in estimating the number of cells in the image.

# Conclusion

As our first fully unsupervised algorithm we ran, we don't have anything to compare it to but the results surprise me as being seemingly good. We need to run more experiments on datasets that match the test dataset. As we noted in the algorithm outline, only half of the steps are needed to do blob and seed detection. Perhaps we can update this pipeline with more up to date algorithms and more appropriate ones for our data and see how it performs.




