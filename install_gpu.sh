#!/bin/bash

# GPU-Optimized Photo Validation - Quick Installation Script
# For Ubuntu with NVIDIA GPU and CUDA 12.0+

set -e

echo "================================================================"
echo "GPU-Optimized Photo Validation Installation"
echo "================================================================"
echo ""

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Please activate your virtual environment first!"
    echo "Run: source venv/bin/activate"
    exit 1
fi

echo "Step 1: Verifying GPU and CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. CUDA toolkit may not be installed."
    echo "Continuing anyway..."
fi

nvidia-smi
echo ""

echo "Step 2: Upgrading pip..."
pip install --upgrade pip
echo ""

echo "Step 3: Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
echo ""

echo "Step 4: Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "Step 5: Installing TensorFlow with CUDA support..."
pip install tensorflow[and-cuda]==2.16.1
echo ""

echo "Step 6: Verifying TensorFlow CUDA..."
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); gpus=tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)}')"
echo ""

echo "Step 7: Installing DeepFace and dependencies..."
pip install deepface==0.0.93 retina-face==0.0.17
echo ""

echo "Step 8: Installing EasyOCR..."
pip install easyocr==1.7.1
echo ""

echo "Step 9: Installing OpenCV..."
pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80
echo ""

echo "Step 10: Installing FastAPI and utilities..."
pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 python-multipart==0.0.20
pip install numpy==1.26.4 Pillow==10.2.0 gdown==5.1.0 mtcnn==0.1.1
echo ""

echo "Step 11: Setting up environment variables..."
# Check if CUDA paths are in bashrc
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
python3 -c "from deepface import DeepFace; print('Downloading Facenet512...'); DeepFace.build_model('Facenet512'); print('Downloading Retinaface...'); DeepFace.build_model('Retinaface'); print('Models ready!')"
echo ""

echo "Step 13: Running GPU verification test..."
cat > /tmp/gpu_test.py << 'EOF'
import tensorflow as tf
import torch

print("\n" + "="*60)
print("GPU VERIFICATION")
print("="*60)

# TensorFlow
print("\n[TensorFlow]")
gpus = tf.config.list_physical_devices('GPU')
print(f"✓ Version: {tf.__version__}")
print(f"✓ GPUs detected: {len(gpus)}")
if gpus:
    print(f"✓ GPU ready: {gpus[0]}")
else:
    print("✗ No GPU detected by TensorFlow")

# PyTorch
print("\n[PyTorch]")
print(f"✓ Version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("✗ No GPU detected by PyTorch")

print("\n" + "="*60)
if len(gpus) > 0 and torch.cuda.is_available():
    print("✓ GPU SETUP SUCCESSFUL!")
else:
    print("✗ GPU setup incomplete - check troubleshooting guide")
print("="*60 + "\n")
EOF

python3 /tmp/gpu_test.py
rm /tmp/gpu_test.py
echo ""

echo "================================================================"
echo "Installation Complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo "1. Test the validation script:"
echo "   python stage_2_response_gpu.py"
echo ""
echo "2. Start the API server:"
echo "   python stage_2_response_api_gpu.py"
echo ""
echo "3. Or with uvicorn:"
echo "   uvicorn stage_2_response_api_gpu:app --host 0.0.0.0 --port 8000 --workers 1"
echo ""
echo "4. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "For troubleshooting, see README_GPU_SETUP.md"
echo "================================================================"
