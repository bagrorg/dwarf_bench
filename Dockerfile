FROM intel/oneapi-basekit

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1


RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    sudo \
    wget \
    --

ARG GIT_SSL_NO_VERIFY=1

ARG oclhelpers_version=0.1.3

RUN mkdir /oclhelpers \
    && cd /oclhelpers \
    && wget https://github.com/kurapov-peter/oclhelpers/releases/download/v${oclhelpers_version}/oclhelpers-v${oclhelpers_version}-Release.tar.gz \
    && tar -xzvf oclhelpers-v${oclhelpers_version}-Release.tar.gz

#Set oclheaders_Dir
ENV oclhelpers_DIR=/oclhelpers/lib/cmake/

RUN apt-get install -y ninja-build libboost-all-dev

# Todo: enable for cuda support
#ENV DPCPP_HOME="/dpcpp"
#RUN mkdir ${DPCPP_HOME} && cd ${DPCPP_HOME} 

#ARG sycl_llvm_sha="f71a1d5c6088c82f2ec0aa1d8b88c19db227d802"
#RUN cd ${DPCPP_HOME} && git clone https://github.com/intel/llvm -b sycl
# RUN python3 ${DPCPP_HOME}/llvm/buildbot/configure.py
# RUN python3 ${DPCPP_HOME}/llvm/buildbot/compile.py

# ENV PATH="${DPCPP_HOME}/llvm/build/bin:${PATH}"
# ENV LD_LIBRARY_PATH="${DPCPP_HOME}/llvm/build/lib:${LD_LIBRARY_PATH}"

RUN apt-get install -y ocl-icd-* opencl-headers

RUN mkdir /cmake && cd /cmake \
    && wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz \
    && tar xf cmake-3.21.0-linux-x86_64.tar.gz \
    && rm cmake-3.21.0-linux-x86_64.tar.gz

ENV PATH="/cmake/cmake-3.21.0-linux-x86_64/bin:${PATH}"

RUN cd root && wget https://github.com/intel/llvm/releases/download/sycl-nightly%2F20210725/dpcpp-compiler.tar.gz \
	    && tar -xf dpcpp-compiler.tar.gz 

RUN cd root && rm -rf dpcpp-compiler.tar.gz

ENV LD_LIBRARY_PATH=/root/dpcpp_compiler/lib:$LD_LIBRARY_PATH
ENV PATH=/root/dpcpp_compiler/bin:$PATH
