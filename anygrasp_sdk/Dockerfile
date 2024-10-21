# Use an NVIDIA CUDA 12.1 # Reverting to 11.8 as 12.1 reaching end of support - base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    python3 \
    python3-pip \
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire project directory
COPY . /app

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the Conda environment from angrasp_env.yml
RUN conda env create -f angrasp_env.yml

# Print Python version
RUN conda run -n anygrasp_sdk python --version

# Initialize Conda for bash shell
RUN /opt/conda/bin/conda init bash

# Activate the Conda environment by default
RUN echo "source activate anygrasp_sdk" >> ~/.bashrc

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Install PyTorch with CUDA 11.1 support
# RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia 
# - c conda-forge

#RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"



# Debug: Print environment information
# RUN echo "Python version:" && python --version && \
    #echo "CUDA version:" && nvcc --version && \
    #echo "PyTorch version:" && python -c "import torch; print(torch.__version__)" && \
    #echo "CUDA available:" && python -c "import torch; print(torch.cuda.is_available())" && \
    #echo "GPU devices:" && python -c "import torch; print(torch.cuda.device_count())"

# Install additional build tools
RUN apt-get update && apt-get install -y \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# The command to run when the container starts
CMD ["bash"]
