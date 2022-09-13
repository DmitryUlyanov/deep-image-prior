FROM nvidia/cuda:11.2.1-devel-ubuntu18.04

# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
    && apt-get clean

# Install python + requirements
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# Clone deep image prior repository
Run ls -alh
RUN git clone https://github.com/NimrodShabtay/deep-image-prior.git
WORKDIR /deep-image-prior
RUN python3 -m pip install -r requirements.txt

# Run script
CMD [ "python3", "./denoising.py"]
