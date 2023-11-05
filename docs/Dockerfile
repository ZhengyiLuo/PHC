# FROM nvcr.io/nvidia/pytorch:21.09-py3
# use 23.02 version which has cuda 12.0.1 with pytorch 1.14 in ubunut 20.4
# see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes
FROM nvcr.io/nvidia/pytorch:23.02-py3

ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs \
 sudo \
 libosmesa6-dev \
 xserver-xephyr \
 && apt-get clean

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym

# Add non-root user and add it to non-password sudoers
RUN useradd --create-home gymuser
RUN adduser gymuser sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER gymuser

# dependency do not conflict pytorch
RUN pip install --upgrade pip &&  pip install --no-cache-dir gdown mujoco numpy-stl vtk patchelf termcolor scikit-image numpy  scipy ipdb 'joblib>=1.2.0' \
 opencv-python==4.6.0.66 \
 tqdm \
 pyyaml \
 wandb \
 scikit-image \
 gym \
 git+https://github.com/ZhengyiLuo/smplx.git@master \
 lxml \
 human_body_prior \
 autograd \
 scikit-learn \
 chumpy \
 wandb \
 pyvirtualdisplay \
 chardet \
 cchardet \
 geoopt \ 
 imageio-ffmpeg \
 easydict \
 open3d \
 torchgeometry \
 pytorch_lightning \
 rl-games==1.1.4 
    

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH "$PATH:/home/gymuser/.local/bin"

# copy gym repo to docker
COPY --chown=gymuser . .

# install gym modules
ENV PATH="/home/gymuser/.local/bin:$PATH"
RUN cd python && pip install -q -e .

