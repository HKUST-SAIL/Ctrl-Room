FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0+PTX"
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt-get install -y \
  curl \
  ca-certificates \
  sudo \
  git \
  bzip2 \
  libx11-6 \
  tmux \
  wget \
  build-essential \
  zsh \
  vim  \
  curl \
  dirmngr \
  gpg\
  rsync \
  ffmpeg \
  libsm6 \
  libxext6 \
  libglib2.0-dev \
  libgles2-mesa-dev \
  openssh-client \
  openssh-server \
  curl \
  ca-certificates \
  sudo \
  bzip2 \
  zip \
  libx11-6 \
  libglfw3-dev \
  libgles2-mesa-dev \
  libglib2.0-0 \
  build-essential \
  curl \
  libegl1-mesa-dev \
  libgl1-mesa-dev \
  libgles2-mesa-dev \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  python-is-python3 \
  python3 \
  python3-pip \
  wget \
  libgl1-mesa-glx \
  libopenmpi-dev \
  && rm -rf /var/lib/apt/lists/*


# set the mount
RUN mkdir /app && mkdir /app/workspace && mkdir /app/libs 

# Upgrade pip
RUN python3 -m pip install --upgrade pip


ENV HOME=/root
RUN mkdir $HOME/.config $HOME/.ssh && chmod -R 0755 $HOME

# set zsh
RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
  && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions


# setup opencv
RUN git clone https://github.com/fangchuan/opencv-3.4.1.git /app/libs/opencv-3.4.1 && cd /app/libs/opencv-3.4.1/ \
  && mkdir build \
  && cd build && sudo apt update && sudo apt-get install -y cmake \
  libgtk2.0-dev \
  pkg-config \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libtbb2 \
  libtbb-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev
RUN cd /app/libs/opencv-3.4.1/build/ && \
  rm -rf * && \
  cmake -D BUILD_SHARED_LIBS=ON  -D WITH_CUDA=OFF -DWITH_OPENEXR=OFF -DBUILD_OPENEXR=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF ../ && \
  make -j8 && \
  sudo make install
RUN sudo rm -rf /app/libs/opencv-3.4.1

# download and install cmake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.tar.gz \
    && tar -xzf cmake-3.19.2-Linux-x86_64.tar.gz -C /opt/ \
    && rm cmake-3.19.2-Linux-x86_64.tar.gz \
    && ln -sf /opt/cmake-3.19.2-Linux-x86_64/bin/* /usr/bin/

# setup open3d
RUN git clone https://github.com/fangchuan/Open3D-0.15.git /app/libs/Open3D && cd /app/libs/Open3D && \
  bash util/install_deps_ubuntu.sh "assume-yes" && \
  mkdir build && \
  cd build && \
  cmake -DBUILD_PYTHON_MODULE=OFF -DBUILD_ISPC_MODULE=OFF ../ \
  && make -j8 && sudo make install
RUN sudo rm -rf /app/libs/Open3D/

# setup glog
RUN git clone https://github.com/fangchuan/glog.git /app/libs/glog && cd /app/libs/glog && \
  cmake -S . -B build -G "Unix Makefiles" && \
  cmake --build build && \
  sudo cmake --build build --target install
RUN sudo rm -rf /app/libs/glog

# setup boost
RUN git clone https://github.com/fangchuan/boost_180.git /app/libs/boost_1_80_0 && cd /app/libs/boost_1_80_0 && \
  ./bootstrap.sh && \
  sudo ./b2 install
RUN sudo rm -rf /app/libs/boost_1_80_0

# setup poissonrecon
RUN git clone https://github.com/fangchuan/PoissonRecon.git /app/libs/PoissonRecon && cd /app/libs/PoissonRecon && \
  make -j8
# RUN rm -rf /app/PoissonRecon/

# setup PanoTexturing
RUN git clone https://github.com/fangchuan/PanoTexturing.git /app/libs/PanoTexturing && cd /app/libs/PanoTexturing && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make -j8

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
  MAMBA_ROOT_PREFIX=/root/micromamba \
  CONDA_PREFIX=/root/micromamba \
  PATH=/root/micromamba/bin:$PATH

# Set up the base Conda environment by installing PyTorch and friends.
COPY settings/conda-linux-64.lock /app/conda-linux-64.lock
RUN micromamba create -qy -n base -f /app/conda-linux-64.lock \
  && rm /app/conda-linux-64.lock \
  && micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX" \
  && micromamba clean -qya

# setup diffu_layout and pano_ctrlnet environment
COPY settings/requirements.txt /app/requirements.txt
# 在单个 bash 会话中执行所有命令
RUN /bin/bash -c "conda create -n ctrlroom python=3.10 -y && \
    source activate ctrlroom && \
    pip install -r /app/requirements.txt"

# copy libstdc++6
RUN sudo cp /root/micromamba/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
# 初始化 conda 并设置默认环境
RUN conda init bash && \
    echo "conda activate ctrlroom" >> ~/.bashrc
# setup ssh
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]