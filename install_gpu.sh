#!/bin/bash

# GPU-Optimized Photo Validation - Installation WITHOUT EasyOCR
# For Ubuntu with NVIDIA GPU and CUDA 12.0+

set -e

echo "================================================================"
echo "GPU-Optimized Photo Validation Installation (NO EASYOCR)"
echo "================================================================"
echo ""

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Please activate your virtual environment first!"
    echo "Run: source venv/bin/activate"
    exit 1
fi

echo "Step 1: Verifying GPU and CUDA..."
nvidia-smi
echo ""

echo "Step 2: Upgrading pip..."
pip install --upgrade pip
echo ""

echo "Step 3: Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
echo ""

echo "Step 4: Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "Step 5: Installing TensorFlow with CUDA support..."
pip install tensorflow[and-cuda]==2.16.1
echo ""

echo "Step 6: Installing tf-keras (required for compatibility)..."
pip install tf-keras==2.16.0
echo ""

echo "Step 7: Verifying TensorFlow CUDA..."
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); gpus=tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)}')"
echo ""

echo "Step 8: Installing DeepFace and dependencies..."
pip install deepface==0.0.93 retina-face==0.0.17
echo ""

echo "Step 9: Installing OpenCV..."
pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80
echo ""

echo "Step 10: Installing FastAPI and utilities..."
pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 python-multipart==0.0.20
pip install numpy==1.26.4 Pillow==10.2.0 gdown==5.1.0 mtcnn==0.1.1
echo ""

echo "Step 11: Setting up environment variables..."
if ! grep -q "CUDA" ~/.bashrc; then
    echo "Adding CUDA paths to ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# CUDA paths for GPU acceleration" >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc
    source ~/.bashrc
fi
echo ""

echo "Step 12: Pre-downloading DeepFace models..."
python3 -c "from deepface import DeepFace; print('Downloading models...'); DeepFace.build_model('Facenet512'); DeepFace.build_model('Retinaface'); print('Models ready!')"
echo ""

echo "================================================================"
echo "Installation Complete! (WITHOUT EasyOCR)"
echo "================================================================"
echo ""
echo "Next steps:"
echo "1. Replace your stage_2_response_gpu.py with stage_2_response_gpu_no_easyocr.py"
echo "2. Test: python stage_2_response_gpu.py"
echo "3. Start API: python stage_2_response_api_gpu.py"
echo ""
echo "================================================================"
