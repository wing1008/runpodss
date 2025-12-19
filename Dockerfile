FROM runpod/base:0.6.0-cuda11.8

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build (optional but recommended)
RUN python -c "from diffusers import MotionAdapter; MotionAdapter.from_pretrained('guoyww/animatediff-motion-adapter-v1-5-2', torch_dtype='auto')"
RUN python -c "from diffusers import AnimateDiffPipeline; AnimateDiffPipeline.from_pretrained('SG161222/Realistic_Vision_V5.1_noVAE', torch_dtype='auto')"

# Copy handler
COPY handler.py /workspace/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf_cache

# RunPod serverless entry
CMD ["python", "-u", "/workspace/handler.py"]
