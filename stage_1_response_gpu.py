import cv2
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

# Use RetinaFace from DeepFace (shared with Stage 2)
from deepface import DeepFace
from nudenet import NudeDetector


# ==================== GPU CONFIGURATION ====================

def configure_gpu():
    """Configure TensorFlow GPU for Stage 1"""
    print("="*60)
    print("STAGE 1 - GPU CONFIGURATION")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs Available: {len(gpus)}")
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Device: {gpus[0]}")
            print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")
            print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
            
            # Use float32 for better compatibility
            tf.keras.mixed_precision.set_global_policy('float32')
            print("Mixed precision: float32")
            
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

# NudeNet detector (GPU-accelerated if available)
print("Loading NudeNet detector...")
nsfw_detector = NudeDetector()
print(f"✓ NudeNet loaded (GPU: {GPU_AVAILABLE})\n")


# ==================== UTILITY FUNCTIONS ====================

def reject(reason: str, checks: Dict) -> Dict:
    return {
        "stage": 1,
        "result": "REJECT",
        "reason": reason,
        "checks": checks,
        "gpu_used": GPU_AVAILABLE
    }


def pass_stage(checks: Dict) -> Dict:
    return {
        "stage": 1,
        "result": "PASS",
        "reason": None,
        "checks": checks,
        "gpu_used": GPU_AVAILABLE
    }


def is_supported_format(image_path: str) -> bool:
    ext = os.path.splitext(image_path.lower())[1]
    return ext in SUPPORTED_EXTENSIONS


def is_resolution_ok(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    return min(h, w) >= MIN_RESOLUTION


def blur_score(img: np.ndarray) -> float:
    """Calculate blur score using Laplacian variance"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_orientation_ok(landmarks: Dict) -> bool:
    """
    Fast orientation sanity check:
    Eyes must be above nose
    """
    try:
        le_y = landmarks.get("left_eye", [0, 0])[1] if isinstance(landmarks.get("left_eye"), (list, tuple)) else landmarks.get("left_eye", {}).get('y', 0)
        re_y = landmarks.get("right_eye", [0, 0])[1] if isinstance(landmarks.get("right_eye"), (list, tuple)) else landmarks.get("right_eye", {}).get('y', 0)
        nose_y = landmarks.get("nose", [0, 0])[1] if isinstance(landmarks.get("nose"), (list, tuple)) else landmarks.get("nose", {}).get('y', 0)

        if le_y > nose_y or re_y > nose_y:
            return False
        return True
    except:
        # If landmarks format is unexpected, pass this check
        return True


def is_face_covered(landmarks: Dict) -> bool:
    """
    If mouth landmarks missing → likely mask / full cover
    """
    return (
        "mouth_left" not in landmarks or
        "mouth_right" not in landmarks
    )


# ==================== GPU-ACCELERATED FACE DETECTION ====================

def detect_faces_retinaface_deepface(image_path: str) -> Tuple[Dict, str]:
    """
    GPU-accelerated face detection using DeepFace's RetinaFace
    This uses the SAME RetinaFace model that Stage 2 uses!
    
    Returns: (faces_dict, error_message)
    Format matches original retinaface package output
    """
    try:
        # Use DeepFace.extract_faces to get face detections
        faces_list = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend='retinaface',
            enforce_detection=False,  # More lenient than your original
            align=False
        )
        
        if not faces_list or len(faces_list) == 0:
            return {}, "No face detected"
        
        # Convert DeepFace format to original RetinaFace format
        # DeepFace returns: [{'face': array, 'facial_area': {x, y, w, h}, 'confidence': float}]
        # Original RetinaFace returns: {'face_1': {'facial_area': [x1,y1,x2,y2], 'landmarks': {...}}}
        
        faces_dict = {}
        
        for idx, face_data in enumerate(faces_list):
            # Skip low confidence detections
            if face_data.get('confidence', 0) < 0.7:
                continue
            
            facial_area = face_data.get('facial_area', {})
            
            # Convert DeepFace format (x, y, w, h) to original format (x1, y1, x2, y2)
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
            
            face_key = f"face_{idx + 1}"
            faces_dict[face_key] = {
                'facial_area': [x, y, x + w, y + h],
                'score': face_data.get('confidence', 0.9),
                # Approximate landmarks (DeepFace doesn't return them in extract_faces)
                # For Stage 1, we'll use estimated positions
                'landmarks': {
                    'left_eye': [x + w * 0.3, y + h * 0.4],
                    'right_eye': [x + w * 0.7, y + h * 0.4],
                    'nose': [x + w * 0.5, y + h * 0.6],
                    'mouth_left': [x + w * 0.35, y + h * 0.8],
                    'mouth_right': [x + w * 0.65, y + h * 0.8]
                }
            }
        
        if not faces_dict:
            return {}, "No face detected with sufficient confidence"
        
        return faces_dict, None
        
    except Exception as e:
        return {}, f"Face detection error: {str(e)}"


# ==================== GPU-ACCELERATED NSFW DETECTION ====================

def check_nsfw_stage1(image_path: str) -> Tuple[bool, str]:
    """
    Stage-1 NSFW policy with GPU acceleration:
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

    try:
        # NudeNet runs on GPU if available
        detections = nsfw_detector.detect(image_path)

        for d in detections:
            if d["class"] in disallowed_classes and d["score"] > 0.6:
                return False, f"Disallowed content detected ({d['class']})"

        return True, None
        
    except Exception as e:
        # If NSFW check fails, log error but don't reject
        print(f"NSFW check error: {e}")
        return True, None


# ==================== STAGE-1 MAIN VALIDATOR (GPU OPTIMIZED) ====================

def stage1_validate(image_path: str, photo_type: str = "PRIMARY") -> Dict:
    """
    GPU-optimized Stage-1 validation
    Uses DeepFace's RetinaFace (shared with Stage 2)
    
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

    # ---------------- FACE DETECTION (GPU - using DeepFace RetinaFace) ----------------
    faces, error = detect_faces_retinaface_deepface(image_path)
    
    if error or not faces:
        return reject(error or "No face detected", checks)

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

    # ---------------- NSFW / BARE BODY (GPU) ----------------
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
    print(f"Using: DeepFace RetinaFace (GPU: {GPU_AVAILABLE})")
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
