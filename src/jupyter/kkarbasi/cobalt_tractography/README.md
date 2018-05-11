# _COBALT Tractography_
Package that performs tractography on 3D tiff stack volumes

## System Requirements
Currently tested on ubuntu 16

### Software Dependencies
The following python packages are needed. They will be automatically downloaded and installed once you pip install the package:

scikit_image <br/>
scipy <br/>
numpy <br/>
requests <br/>
intern <br/>
tifffile<br/>
matplotlib <br/>
scikit_learn <br/>
scikit-fmm <br/>

## Installation
run ``` pip install .``` inside cobalt_tractography directory to install the package and its requirements

## Use
You can now use the functions by importing the following in your python script:
```
from cobalt_tractography.bossHandler import *
from cobalt_tractography.tractography import *
```


