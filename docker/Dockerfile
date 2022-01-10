ARG BASE_IMAGE=ubuntu:20.04
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE} as dev-base

ENV TZ="Europe/Berlin"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build && \
    /opt/conda/bin/conda install -y nomkl pyyaml numpy ipython ninja setuptools cmake cffi typing future && \
    /opt/conda/bin/conda clean -ya

RUN conda install typing_extensions

WORKDIR /root

# Install Torch and TorchVision
# CPU version
#RUN pip3 install torch torchvision
# CPU and GPU version
RUN pip3 install --trusted-host download.pytorch.org \
    torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 \
    -f https://download.pytorch.org/whl/cu102/torch_stable.html 

# Install prerequirement for installing Swift
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        attr \
        liberasurecode1 \
        liberasurecode-dev \
        memcached \
        rsyslog \
        rsync \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install supervisor pytz

RUN pip3 install dnspython \
        eventlet \
        greenlet \
        lxml \
        netifaces \
        pastedeploy \
        pip \
        pyeclib \
        setuptools \
        simplejson \
        xattr \ 
        iperf3 \
        psutil

# Overcome the proxy issue
ENV GIT_SSL_NO_VERIFY=1

# Install Swift
RUN apt-get update && \
    apt-get install -y --no-install-recommends git-core && \
    git clone --branch 3.11.1 --single-branch --depth 1 https://github.com/openstack/python-swiftclient.git /usr/local/src/python-swiftclient && \
    cd /usr/local/src/python-swiftclient && python3 setup.py develop && \
    git clone --branch 2.27.0 --single-branch --depth 1 https://github.com/openstack/swift.git /usr/local/src/swift && \
    cd /usr/local/src/swift && python3 setup.py develop && \
    apt-get remove -y --purge git-core git && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
