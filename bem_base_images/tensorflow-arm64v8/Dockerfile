### Instructions from here: https://qengineering.eu/install-tensorflow-2.4.0-on-raspberry-64-os.html
### https://qengineering.eu/install-ubuntu-20.04-on-raspberry-pi-4.html
### All available wheels are listed and linked here: https://github.com/Qengineering/TensorFlow-Raspberry-Pi_64-bit

FROM ubuntu:focal

RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
# Update pip
RUN python3 -m pip install --upgrade pip

# Install TF dependencies
RUN apt-get -y install gfortran
# Note: setting these env variables is needed to prevent installation get stuck with tzdata
# (https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image)
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install libhdf5-dev libc-ares-dev libeigen3-dev
RUN apt-get -y install libatlas-base-dev libopenblas-dev libblas-dev
RUN apt-get -y install liblapack-dev

# Copy already buit wheels for packages that are only available as source code via pip
COPY dep-wheels /dep-wheels
RUN pip install --upgrade setuptools
# TF 2.4.1 requires numpy==1.19.*
RUN pip install -f /dep-wheels numpy~=1.19.2
RUN pip install -f /dep-wheels pybind11==2.6.2
RUN pip install -f /dep-wheels Cython==0.29.21
# h5py must be build with numpy~=1.19.2
# No wheel for h5py available, only source code -> building wheel takes forever -> load from local pre-built wheel
RUN pip install -f /dep-wheels h5py==2.10.0
# Install gdown for download from Google Drive
RUN pip install gdown

RUN mkdir /tf-wheel
# Download the wheel for TF 2.4.1 and Ubuntu 20.04 (229 MB)
RUN gdown -O /tf-wheel/tensorflow-2.4.1-cp38-cp38-linux_aarch64.whl https://drive.google.com/uc?id=1rfgF2U2oZJvQSMbGNZl8f5jbWP4fY6UW
# Install TF 2.4.1 (search for dependency grpcio in dep-wheels)
RUN pip install /tf-wheel/tensorflow-2.4.1-cp38-cp38-linux_aarch64.whl -f /dep-wheels

# Collect all wheels from pip cache (these are the ones without a downloadable wheel and can take forever
# to built! -> Store them for later re-use if necessary)
RUN cd /root/.cache/pip && find -iname '*.whl' -exec cp {} /dep-wheels \;






