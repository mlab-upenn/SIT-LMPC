FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

# apt packages
RUN apt update

RUN apt update \
    && apt install -y git \
    automake \
    build-essential \
    curl \
    nano \
    tmux \
    wget \
    unzip \
    zip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/conda 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/conda/miniconda.sh 
RUN bash /opt/conda/miniconda.sh -b -p /opt/miniconda 
RUN /opt/miniconda/bin/conda init bash
ENV PATH="/opt/miniconda/bin:${PATH}"

SHELL ["/bin/bash", "-i", "-c"] 

RUN conda create -n py311 python=3.11.9
RUN conda activate py311

SHELL ["conda", "run", "-n", "py311", "/bin/bash", "-c"]

# RUN mkdir /workspace/dockersetup
# RUN cd /workspace/dockersetup
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# RUN python3 get-pip.py
# COPY ./requirements.txt /workspace/dockersetup/requirements.txt

# # pip packages
# RUN pip install -r /workspace/dockersetup/requirements.txt
# RUN pip install git+https://github.com/VLL-HD/FrEIA.git

RUN conda install -n base ipykernel --update-deps --force-reinstall
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-opencv

# from f1tenth_gym
ENV LIBGL_ALWAYS_INDIRECT=1
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update --fix-missing && \
    apt-get install -y \
                    python3-dev \
                    python3-pip \
                    git \
                    build-essential \
                    libgl1-mesa-dev \
                    mesa-utils \
                    libglu1-mesa-dev \
                    fontconfig \
                    libfreetype6-dev

# RUN pip3 install PyOpenGL \
#                  PyOpenGL_accelerate

RUN pip install -U "jax[cuda12]"
RUN pip install flax optax distrax pynput tqdm
RUN pip install matplotlib opencv-python PyQt6

# # RUN useradd -m -d /home/user -u 1004 user
# # USER root
# # RUN chown -R user:user /workspace
# # USER 1004:1004
# # ENV PATH="${PATH}:/home/user/.local/bin"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
