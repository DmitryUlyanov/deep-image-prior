FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
    && apt-get clean

# Install python + requirements
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
# Clone deep image prior repository

RUN python3 -m pip install Pillow==8.2.0
Run python3 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
Run ls -lah
Run ls -lah
RUN git clone https://github.com/NimrodShabtay/deep-image-prior.git
WORKDIR /deep-image-prior
RUN python3 -m pip install -r requirements.txt

RUN export LC_ALL=C.UTF-8 && export LANG=C.UTF-8

# Run script
CMD [ "python3", "./denoising_example.py"]
