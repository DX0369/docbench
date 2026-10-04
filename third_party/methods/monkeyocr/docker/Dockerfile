# Use CUDA environment and install basic dependencies
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 git curl \
    wget poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Force create symbolic links (overwrite if exists)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Configure pip to use faster mirror and upgrade
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip setuptools || echo "Warning: pip upgrade failed"
ENV PIP_NO_CACHE_DIR=1

# Install PyTorch (updated to version 2.6.0)
ARG CUDA_VERSION=126

# Copy project code
COPY --chown=appuser:appuser . /app/MonkeyOCR

# Set working directory to /app/MonkeyOCR
WORKDIR /app/MonkeyOCR

# Install model download related dependencies
RUN pip install modelscope huggingface_hub

# Install PaddlePaddle and PaddleX for PP-DocLayoutV2 support
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/
RUN pip install paddlex[base]==3.3.0
RUN pip install langchain==0.3.26

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu${CUDA_VERSION}

# Install project itself
RUN pip install -e .

# Install inference backend dependencies
RUN pip install lmdeploy==0.9.2

# Copy model download scripts
COPY --chown=appuser:appuser docker/download_models.sh /app/MonkeyOCR/
COPY --chown=appuser:appuser docker/entrypoint.sh /app/MonkeyOCR/
RUN chmod +x /app/MonkeyOCR/download_models.sh /app/MonkeyOCR/entrypoint.sh

# Create model directory
RUN mkdir -p /app/MonkeyOCR/model_weight && chown -R appuser:appuser /app/MonkeyOCR/model_weight

# Apply LMDeploy patcher
ARG LMDEPLOY_PATCHED=false
RUN if [ "$LMDEPLOY_PATCHED" = "true" ]; then \
      python /app/MonkeyOCR/tools/lmdeploy_patcher.py patch; \
      echo "Successfully apply lmdeploy patch" \
    else \
      echo "LMDEPLOY_PATCHED is false, skipping patch"; \
    fi

# Switch to non-root user
USER appuser

# Default command: start through entrypoint script
CMD ["/app/MonkeyOCR/entrypoint.sh"]
