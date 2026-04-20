# KayTrain Worker — RunPod Serverless
# Base: PyTorch 2.1 + CUDA 11.8 + Python 3.10
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download R(2+1)D-18 weights to avoid cold-start delay
RUN python3 -c "import torchvision; torchvision.models.video.r2plus1d_18(weights='DEFAULT'); print('Weights cached OK')"

# Copy handler
COPY handler.py .

# Start serverless worker
CMD ["python3", "-u", "handler.py"]
