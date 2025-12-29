# GPU-ready base with CUDA + cuDNN (no need to install cuDNN manually)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# **Optional but useful**: same TMPDIR trick as your script
ENV TMPDIR=/workspace/tmp
RUN mkdir -p /workspace/tmp

# System deps: Python + ffmpeg + audio + git + wget + nodejs
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Workdir in container
WORKDIR /workspace/FATCR

# Copy your entire project (including data/processed KB)
COPY . .

# Python deps (equivalent of your venv+pip installs, but global)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install faiss-cpu

# Expose Streamlit port
EXPOSE 8501

# Same as final step of start_factr.sh
CMD ["streamlit", "run", "factr_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]