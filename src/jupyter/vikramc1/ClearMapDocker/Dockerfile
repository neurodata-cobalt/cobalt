# Start from a clean ubuntu 16.04 install
FROM ubuntu:xenial

# build apt-get packages and install needed packages
RUN apt-get update && apt-get -y install python-opencv \
    cython \
    python-tifffile \
    python-h5py \
    python-natsort \
    spyder \
    git \
    python-pip \
    vim \
    ocl-icd-opencl-dev

# install python dependencies
RUN pip install --upgrade pip && pip install scikit-image jupyter ipyvolume

# download ClearMap
RUN git clone https://github.com/ChristophKirst/ClearMap.git

# copy the settings.py file with the correct path locations
COPY Settings.py /ClearMap/ClearMap/

# install ClearMap and move folders to appropriate locations
RUN cd ClearMap/ &&  python setup.py install && cp -r ClearMap/Data/ /usr/local/lib/python2.7/dist-packages/ClearMap-0.9.2-py2.7-linux-x86_64.egg/ClearMap/Data/ 

# set LD_LIBRARY_PATH to default 
ENV LD_LIBRARY_PATH=/usr/local/lib:/mnt/data/elastix/lib/ 

WORKDIR /run 

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
#CMD ["jupyter", "notebook"]
