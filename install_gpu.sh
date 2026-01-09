#!/bin/bash

# GPU-Optimized Photo Validation - FIXED Installation Script
# Resolves TensorFlow version conflicts

set -e

echo "================================================================"
echo "GPU-Optimized Photo Validation Installation (FIXED)"
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

nvidia-smi
echo ""

echo "Step 2: Upgrading pip..."
pip install --upgrade pip
echo ""

echo "Step 3: Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
echo ""

echo "Step 4: Installing TensorFlow 2.16.1 with CUDA support..."
# Use TensorFlow 2.16.1 which works better with the ecosystem
pip install tensorflow[and-cuda]==2.16.1
echo ""

echo "Step 5: Installing tf-keras (required for compatibility)..."
pip install tf-keras==2.16.0
echo ""

echo "Step 6: Verifying TensorFlow CUDA..."
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); gpus=tf.config.list_physical_devices('GPU'); print(f'GPUs: {len(gpus)}')"
echo ""

echo "Step 7: Installing DeepFace and dependencies..."
pip install deepface==0.0.93 retina-face==0.0.17
echo ""

echo "Step 8: Installing NudeNet..."
pip install nudenet==3.4.2
echo ""

echo "Step 9: Installing OpenCV..."
pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80
echo ""

echo "Step 10: Installing FastAPI and utilities..."
pip install fastapi==0.115.6 uvicorn[standard]==0.34.0 python-multipart==0.0.20
echo ""

echo "Step 11: Installing other dependencies..."
pip install numpy==1.26.4 Pillow==10.2.0 gdown==5.1.0 mtcnn==0.1.1
echo ""

echo "Step 12: Setting up environment variables..."
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

echo "Step 13: Pre-downloading DeepFace models..."
python3 -c "from deepface import DeepFace; print('Downloading Facenet512...'); DeepFace.build_model('Facenet512'); print('Downloading Retinaface...'); DeepFace.build_model('Retinaface'); print('Models ready!')"
echo ""

echo "Step 14: Running GPU verification test..."
cat > /tmp/gpu_test.py << 'EOF'
import tensorflow as tf
import torch
from retinaface import RetinaFace
from deepface import DeepFace

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

# RetinaFace
print("\n[RetinaFace]")
print("✓ RetinaFace imported successfully")

# DeepFace
print("\n[DeepFace]")
print("✓ DeepFace imported successfully")

print("\n" + "="*60)
if len(gpus) > 0 and torch.cuda.is_available():
    print("✓ GPU SETUP SUCCESSFUL!")
else:
    print("✗ GPU setup incomplete")
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
echo ""
echo "1. Download a test image:"
echo "   wget https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg -O Fullface.jpeg"
echo ""
echo "2. Test Stage 1:"
echo "   python3 stage_1_response_gpu.py Fullface.jpeg PRIMARY"
echo ""
echo "3. Test Stage 2:"
echo "   python3 stage_2_response_gpu.py"
echo ""
echo "4. Start Stage 1 API:"
echo "   uvicorn stage_1_response_api_gpu:app --host 0.0.0.0 --port 8001 --workers 1"
echo ""
echo "5. Start Stage 2 API:"
echo "   uvicorn stage_2_response_api_gpu:app --host 0.0.0.0 --port 8000 --workers 1"
echo ""
echo "6. Test the APIs:"
echo "   curl http://localhost:8001/health"
echo "   curl http://localhost:8000/health"
echo ""
echo "================================================================"
