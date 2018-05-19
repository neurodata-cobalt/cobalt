FROM ubuntu:16.04
# FROM python:3.6.5

RUN apt-get update
RUN apt-get -y install python-pip
RUN apt-get install build-essential libssl-dev libffi-dev python-dev
RUN apt-get -y install git vim
WORKDIR "/home"
RUN git clone https://github.com/neurodata-cobalt/ndtractography.git
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scikit-image scikit-learn scipy matplotlib tifffile scikit-fmm intern pandas
# RUN pip install -r ndtractography/requirements.txt
WORKDIR "/home/ndtractography"
RUN pip install .
RUN pip install jupyter
RUN jupyter notebook --generate-config --allow-root
# ADD jupyter_notebook_config.py /home/.jupyter/
RUN sed -i "1ic.NotebookApp.port = 8080" ~/.jupyter/jupyter_notebook_config.py \
    && sed -i "1ic.NotebookApp.open_browser = False" ~/.jupyter/jupyter_notebook_config.py \
        && sed -i "1ic.NotebookApp.ip = '*'" ~/.jupyter/jupyter_notebook_config.py
WORKDIR "/home"
# ENTRYPOINT ["jupyter notebook --allow-root"]
