"""
Stage 1 Response GPU - CLEANED VERSION
Fixed imports and removed OCR dependencies
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

# Use RetinaFace from retinaface library
from retinaface import RetinaFace
from deepface import DeepFace
from nudenet import NudeDetector


# ==================== GPU CONFIGURATION ====================

def configure_gpu():
    """Configure TensorFlow to use GPU efficiently"""
    print("="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs Available: {len(gpus)}")
    
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set visible devices
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Device: {gpus[0]}")
            print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")
            
            # Check CUDA and cuDNN
            print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
            print(f"GPU is being used: {tf.test.is_gpu_available(cuda_only=True)}")
            
            # Set mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (float16)")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("WARNING: No GPU detected. Running on CPU.")
    
    print("="*60 + "\n")
    return len(gpus) > 0

# Configure GPU at module load
GPU_AVAILABLE = configure_gpu()


# ==================== CONFIGURATION ====================
MIN_RESOLUTION = 360          # SOP minimum
MIN_FACE_SIZE = 120           # px
BLUR_REJECT = 35              # Laplacian variance threshold

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"
}

# NudeNet detector (loaded once)
nsfw_detector = NudeDetector()


# UTILITY FUNCTIONS

def reject(reason, checks):
    return {
        "stage": 1,
        "result": "REJECT",
        "reason": reason,
        "checks": checks,
        "gpu_used": GPU_AVAILABLE
    }

def pass_stage(checks):
    return {
        "stage": 1,
        "result": "PASS",
        "reason": None,
        "checks": checks,
        "gpu_used": GPU_AVAILABLE
    }

def is_supported_format(image_path):
    ext = os.path.splitext(image_path.lower())[1]
    return ext in SUPPORTED_EXTENSIONS

def is_resolution_ok(img):
    h, w = img.shape[:2]
    return min(h, w) >= MIN_RESOLUTION

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_orientation_ok(landmarks):
    """
    Fast orientation sanity:
    Eyes must be above nose
    """
    le_y = landmarks["left_eye"][1] 
    re_y = landmarks["right_eye"][1]
    nose_y = landmarks["nose"][1]

    if le_y > nose_y or re_y > nose_y:
        return False
    return True

def is_face_covered(landmarks):
    """
    If mouth landmarks missing → likely mask / full cover
    """
    return (
        "mouth_left" not in landmarks or
        "mouth_right" not in landmarks
    )


# NSFW / BARE BODY (STAGE-1)

def check_nsfw_stage1(image_path):
    """
    Stage-1 NSFW policy:
    - ANY nudity / bare body → REJECT
    - No suspend here (per requirement)
    """

    disallowed_classes = {
        # Explicit nudity
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",

        # Bare / semi-nude / inappropriate
        "MALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "UNDERWEAR",
        "SWIMWEAR"
    }

    detections = nsfw_detector.detect(image_path)

    for d in detections:
        if d["class"] in disallowed_classes and d["score"] > 0.6:
            return False, f"Disallowed content detected ({d['class']})"

    return True, None

# STAGE-1 MAIN VALIDATOR

def stage1_validate(image_path, photo_type="PRIMARY"):
    """
    photo_type: PRIMARY | SECONDARY
    """

    checks = {}

    # ---------------- IMAGE READ ----------------
    img = cv2.imread(image_path)
    if img is None:
        return reject("Invalid or unreadable image", checks)
    checks["image_read"] = "PASS"

    # ---------------- FORMAT ----------------
    if not is_supported_format(image_path):
        return reject("Unsupported image format", checks)
    checks["format"] = "PASS"

    # ---------------- RESOLUTION ----------------
    if not is_resolution_ok(img):
        return reject("Low resolution image", checks)
    checks["resolution"] = "PASS"

    # ---------------- FACE DETECTION ----------------
    faces = RetinaFace.detect_faces(image_path)
    if not faces:
        return reject("No face detected", checks)

    if photo_type == "PRIMARY" and len(faces) > 1:
        return reject("Group photo not allowed as primary photo", checks)

    checks["face_count"] = "PASS"

    # Pick first face (Stage-1 does not rank faces)
    face = list(faces.values())[0]
    area = face["facial_area"]
    landmarks = face["landmarks"]

    fw = area[2] - area[0]
    fh = area[3] - area[1]

    if min(fw, fh) < MIN_FACE_SIZE:
        return reject("Face too small or unclear", checks)

    checks["face_size"] = "PASS"

    # ---------------- BLUR ----------------
    blur = blur_score(img)
    if blur < BLUR_REJECT:
        return reject("Image is too blurry", checks)

    checks["blur"] = "PASS"

    # ---------------- ORIENTATION ----------------
    if not is_orientation_ok(landmarks):
        return reject("Improper image orientation", checks)

    checks["orientation"] = "PASS"

    # ---------------- MASK / FACE COVER ----------------
    if is_face_covered(landmarks):
        return reject("Face is covered or wearing a mask", checks)

    checks["face_cover"] = "PASS"

    # ---------------- NSFW / BARE BODY ----------------
    nsfw_ok, nsfw_reason = check_nsfw_stage1(image_path)
    if not nsfw_ok:
        return reject(nsfw_reason, checks)

    checks["nsfw"] = "PASS"

    # ---------------- FINAL ----------------
    return pass_stage(checks)

# ==================== BATCH PROCESSING (GPU OPTIMIZED) ====================

def stage1_validate_batch(
    image_paths: List[str],
    photo_types: List[str] = None
) -> List[Dict]:
    """
    Batch validate multiple images (GPU optimized)
    
    Args:
        image_paths: List of image file paths
        photo_types: List of photo types (PRIMARY/SECONDARY), one per image
    
    Returns:
        List of validation results
    """
    if photo_types is None:
        photo_types = ["PRIMARY"] * len(image_paths)
    
    if len(photo_types) != len(image_paths):
        raise ValueError("photo_types length must match image_paths length")
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"BATCH VALIDATION: {len(image_paths)} images")
    print(f"Using: RetinaFace (GPU: {GPU_AVAILABLE})")
    print(f"{'='*60}\n")
    
    for idx, (img_path, photo_type) in enumerate(zip(image_paths, photo_types)):
        print(f"[{idx+1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
        
        try:
            result = stage1_validate(img_path, photo_type)
            result["image_path"] = img_path
            result["image_index"] = idx
            results.append(result)
            
            print(f"    Result: {result['result']}")
            if result['result'] == 'REJECT':
                print(f"    Reason: {result['reason']}")
            print()
            
        except Exception as e:
            results.append({
                "stage": 1,
                "result": "REJECT",
                "reason": f"Processing error: {str(e)}",
                "checks": {},
                "gpu_used": GPU_AVAILABLE,
                "image_path": img_path,
                "image_index": idx
            })
            print(f"    Result: REJECT (Error: {str(e)})\n")
    
    # Summary
    passed = sum(1 for r in results if r['result'] == 'PASS')
    rejected = len(results) - passed
    
    print(f"{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Rejected: {rejected}")
    print(f"GPU Used: {GPU_AVAILABLE}")
    print(f"{'='*60}\n")
    
    return results


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        image_path = sys.argv[1]
        photo_type = sys.argv[2] if len(sys.argv) > 2 else "PRIMARY"
        
        print(f"\nValidating: {image_path}")
        print(f"Photo Type: {photo_type}\n")
        
        result = stage1_validate(image_path, photo_type=photo_type)
        
        print("\n" + "="*60)
        print("VALIDATION RESULT")
        print("="*60)
        print(f"Stage: {result['stage']}")
        print(f"Result: {result['result']}")
        print(f"GPU Used: {result['gpu_used']}")
        if result['reason']:
            print(f"Reason: {result['reason']}")
        print(f"\nChecks Passed:")
        for check, status in result['checks'].items():
            print(f"  {check}: {status}")
        print("="*60)
    else:
        # Test with default image
        print("Testing with 'Fullface.jpeg'...\n")
        result = stage1_validate("Fullface.jpeg", photo_type="PRIMARY")
        print(result)
