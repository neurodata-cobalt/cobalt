# Cobalt Sprint 2 Goals

##### Team: Vikram Chandrashekhar, Srivathsa PV, Jason Yim, Jonathan Liu

Our sprint 2 goals and assignees are as follows:

1. Registration package with affine and LDDMM registration -- `@vikram`
2. Simple blob detection python package with 1 algorithm (including region-based analysis -- dependent on registration) -- `@srivathsa`
3. demonstrate MVP to reduce gridding artifacts in COLM/La Vision data -- `@jyim`
4. work with `avatr` to help integrate our pipeline into their infrastructure / annotate cell/tractography subvolumes -- `@jonl1096`


___
# Details:


1. Registration package - we will have a complete registration package that includes both affine and LDDMM registration, and evaluation. To begin, I will evaluate daniel's registration of Control9 and show to ailey. If registration meets spec, then will be tested on all `Insula*` 🧠 on the boss and shown to Brian for evaluation. Code will be made available on [Github](https://github.com/neurodata/ndreg) and [Docker Hub](https://hub.docker.com/r/neurodata/ndreg/).

*registration package must meet both quantitative and qualitative standards established on both >= 2LaVision and >= 1 COLM datasets.  This will therefore require obtaining fiducials on each of those brains, and having confidence that those fiducials are right.  Thus, a workflow for generating fiducials must be developed, possibly in collaboration with AVATR. The registered atlas must be overlaid on the brain in native resolution and ndviz links provided with overlays. each week, report on both qualitative and quantitative improvements of performance.*


2. cell detection package - we will include 1 unsupervised cell counting method that works based on our assessment of the existing methods. We intend to write a summary of the existing methods and their performance (assessed using `blob-metrics`) against our 10 annotated subvolumes starting from the simplest methods (binarization + counting, LoG, DoG, HoG). If/when we show that these methods are less effective, we can optimize our current HDoG algorithm by improving clustering methods used. The cell detection package will use outputs from the registration in order to perform region-based analysis which includes: cell count by region, intensity by region, cell density by region, intensity density by region, etc. Code will be made available on [Github](https://github.com/NeuroDataDesign/bloby).

*There is no need for a summary, we need the simplest possibly method that works sufficiently well.  We therefore need qualitative and quantitative evaluation.  The quantitative evaluation must be using >=2 different ROIs on each brain that is registered above (arbitrary subvolumes are insufficient). This means criteria for success must be defined now, what FP & FN rates are acceptable? At a minimum, the ranking of ROIs must be correct, insofar as if one ROI truly has more cells than another, than your code must have the property (some fraction of the time for you to determine).  The manual and automatic annotations must be ingested into the boss, and ndviz links provided showing overlays of them all. Each week, report on both qualitative and quantitative improvements of performance.*  


3. artifact reduction - we will demonstrate an MVP that corrects the bias field per tile in the COLM data. we will begin by surveying methods that exist to correct the bias and begin by trying something simple. Once we have a bias correction method that works we will apply it to all tiles and use terrastitcher to recombine the tiles. The resulting image will be uploaded to the boss to compare to the uncorrected image for qualitative assessment.

*Starting simple is a great idea.  You need a metric and criterion for success.  How many brains? Which brains? Will the new brains be re-registered to the atlas? Each week, report on both qualitative and quantitative improvements of performance.*



4. work with avatr - provide `avatr` with necessary support to integrate our code with their infrastructure. If there isn't a clear weekly goal with avatr, we will spend that week determining how to annotate axons for tractography evaluation or annotating additional cell subvolumes. For all cell volumes annotated with centroids from now on, the centroids will have a consistent format. Centroids will be stored as a csv with the following spec: The first line will be: ```,z,y,x```. Every line after will have the same format where the first column is annotation number and the `z, y, x` coordinates assuming the origin is `0,0,0` where the *top left* of each slice is `x = 0, y = 0` and the *first slice* is `z = 0`.


*This is poorly scoped.  What precisely is the DoD?  AVATR will support what functionality?  The output of cell detection will be ingested and viz links provided?  What about the registration?  AVATR should support that too i would think? And the re-ingest of the bias corrected data? Specify concrete goals and DoDs so we can evaluate effectively. Each week, report on both qualitative and quantitative improvements of performance.*

_____

# Definition of Done

## Registration package

### Goals

* Qualitatively evaluate at least 2 La Vision and 1 COLM registration by uploading to the Boss
   * ideally this is done at native resolution
* Quantitatively evaluate at least 2 La Vision and 1 COLM registration by computing fiducial errors 
* Generate a workflow for getting fiducials on raw data and document this workflow in a Google Doc


## Cell Detection Package

### Goals

* Run LoG on the existing annotated subvolumes with a manually chosen scale that works well for most of the data and do quantitative and qualitative evaluation. 
* Use the cell count results of LoG to obtain region wise cell counts and compare this with annotated cell counts/region.
* With LoG as the detection algorithm, document the package with usage instructions.
* Make the package pip installable

### Application to Real Data

The comparison and evaluation is being done on _ailey-dev_ - _S3617_, _Atenolol2_, and _ISO1_ experiments:
* Qualitative evaluation of LoG will be done by uploading the detected results and the annotated results to Boss. 
* Quantitative evaluation will be done using the _blob-metrics_ package using accuracy, precision, recall and f-measure.

### Importance

This will complete a well documented cell detection package with LoG as the detection algorithm with qualititative and quantitative evaluation.


## Tractography annotations and evaluation

### Goals

* Generate manual annotations for 5 tractography subvolumes.
* Implement a tractography evaluation method/package.
    * The evaluation metric we will start with is the difference between the integrated distance of the manual annotations and machine annotations in each ROI of a registered brain.
    
### Importance

Since we plan on implementing an unsupervised tractography package, we will need an annotated dataset and evaluation method to test against.





