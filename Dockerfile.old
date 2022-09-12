FROM nvidia/cuda:9.0-cudnn7-devel

# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
    && apt-get clean

# Install python miniconda3 + requirements
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
RUN curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
    && rm Miniconda3-latest-Linux-x86_64.sh
COPY environment.yml environment.yml
RUN conda env update -n=root --file=environment.yml
RUN conda clean -y -i -l -p -t && \
    rm environment.yml

# Clone deep image prior repository
RUN git clone https://github.com/DmitryUlyanov/deep-image-prior.git
WORKDIR /deep-image-prior

# Start container in notebook mode
CMD jupyter notebook --ip="*" --no-browser --allow-root

