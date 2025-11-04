# 使用NVIDIA Isaac Sim基础镜像
FROM nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1

# 设置环境变量
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV DEBIAN_FRONTEND=noninteractive
ENV ENV_NAME=NavRL

# 设置工作目录
WORKDIR /workspace

# 安装miniconda
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 将conda添加到PATH
ENV PATH /opt/conda/bin:$PATH

# 配置conda
RUN conda init bash && \
    conda config --set always_yes yes --set changeps1 no

# 复制您的代码到容器中
COPY . /workspace/

# 设置conda环境并安装依赖
RUN conda create -n $ENV_NAME python=3.10 -c conda-forge -y && \
    echo "conda activate $ENV_NAME" >> ~/.bashrc

# 初始化conda并激活环境
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    # 安装基础依赖
    pip install numpy==1.26.4 && \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
    pip install "pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2" && \
    pip install imageio-ffmpeg==0.4.9 && \
    pip install moviepy==1.0.3 && \
    pip install hydra-core --upgrade && \
    pip install einops && \
    pip install pyyaml && \
    pip install rospkg && \
    pip install matplotlib && \
    # 安装TensorDict依赖
    pip uninstall -y tensordict && \
    pip install tomli && \
    # 安装TensorDict
    cd /workspace/isaac-training/third_party/tensordict && \
    python setup.py develop && \
    # 安装TorchRL
    cd /workspace/isaac-training/third_party/rl && \
    python setup.py develop && \
    # 检查torch安装
    python -c "import torch; print('Torch version:', torch.__version__); print('Torch path:', torch.__path__)"

# 设置默认的conda环境
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $ENV_NAME" >> ~/.bashrc

# 设置入口点
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate NavRL && \"$@\"", "--"]

# 默认命令
CMD ["bash"]