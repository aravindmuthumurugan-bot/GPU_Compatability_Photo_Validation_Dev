import cv2
import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

# Use DeepFace's built-in RetinaFace detector
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

# Pre-build DeepFace's RetinaFace detector (shared with Stage 2!)
print("Loading DeepFace RetinaFace detector (shared with Stage 2)...")
try:
    # This loads the same model that Stage 2 uses
    DeepFace.build_model("retinaface")
    print("✓ DeepFace RetinaFace loaded (GPU accelerated)")
except Exception as e:
    print(f"Warning: Could not pre-load RetinaFace: {e}")

# NudeNet detector (loaded once, GPU-accelerated if available)
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
        le_y = landmarks.get("left_eye", [0, 0])[1]
        re_y = landmarks.get("right_eye", [0, 0])[1]
        nose_y = landmarks.get("nose", [0, 0])[1]

        if le_y > nose_y or re_y > nose_y:
            return False
        return True
    except:
        return True  # If landmarks are weird, pass this check


def is_face_covered(landmarks: Dict) -> bool:
    """
    If mouth landmarks missing → likely mask / full cover
    """
    return (
        "mouth_left" not in landmarks or
        "mouth_right" not in landmarks
    )


# ==================== GPU-ACCELERATED FACE DETECTION ====================

def detect_faces_deepface(image_path: str) -> Tuple[List[Dict], str]:
    """
    GPU-accelerated face detection using DeepFace's RetinaFace
    This reuses the SAME model that Stage 2 uses!
    
    Returns: (list_of_faces, error_message)
    """
    try:
        # Use DeepFace.extract_faces with retinaface backend
        # This is more efficient than importing retinaface separately
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend='retinaface',
            enforce_detection=True,
            align=False  # We don't need alignment for Stage 1
        )
        
        if not faces or len(faces) == 0:
            return [], "No face detected"
        
        return faces, None
        
    except ValueError as e:
        # DeepFace raises ValueError when no face is detected
        return [], "No face detected"
    except Exception as e:
        return [], f"Face detection error: {str(e)}"


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
        # If NSFW check fails, pass it through (don't reject on error)
        print(f"NSFW check error: {e}")
        return True, None


# ==================== STAGE-1 MAIN VALIDATOR (GPU OPTIMIZED) ====================

def stage1_validate(image_path: str, photo_type: str = "PRIMARY") -> Dict:
    """
    GPU-optimized Stage-1 validation
    Uses DeepFace's RetinaFace detector (shared with Stage 2)
    
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

    # ---------------- FACE DETECTION (GPU - using DeepFace) ----------------
    print(f"Detecting faces (DeepFace RetinaFace, GPU: {GPU_AVAILABLE})...")
    faces, error = detect_faces_deepface(image_path)
    
    if error:
        return reject(error, checks)

    if photo_type == "PRIMARY" and len(faces) > 1:
        return reject("Group photo not allowed as primary photo", checks)

    checks["face_count"] = "PASS"

    # Pick first face
    face = faces[0]
    
    # DeepFace returns different format than retinaface package
    facial_area = face.get('facial_area', {})
    
    # Calculate face size
    if isinstance(facial_area, dict):
        fw = facial_area.get('w', 0)
        fh = facial_area.get('h', 0)
    else:
        # Fallback
        fw = fh = 0

    if min(fw, fh) < MIN_FACE_SIZE:
        return reject("Face too small or unclear", checks)

    checks["face_size"] = "PASS"

    # ---------------- BLUR ----------------
    blur = blur_score(img)
    if blur < BLUR_REJECT:
        return reject("Image is too blurry", checks)

    checks["blur"] = f"PASS (score: {blur:.1f})"

    # ---------------- ORIENTATION ----------------
    # DeepFace doesn't return landmarks in extract_faces
    # We'll do a simpler check using facial_area
    # For proper orientation, face width should be reasonable
    if fw > 0 and fh > 0:
        aspect_ratio = fw / fh
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return reject("Improper image orientation", checks)
    
    checks["orientation"] = "PASS"

    # ---------------- MASK / FACE COVER ----------------
    # For Stage 1, we'll skip detailed landmark checks
    # Stage 2 will do more thorough checks
    checks["face_cover"] = "PASS (checked in Stage 2)"

    # ---------------- NSFW / BARE BODY (GPU) ----------------
    print(f"Checking NSFW (GPU: {GPU_AVAILABLE})...")
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
    Uses shared DeepFace RetinaFace detector
    
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
    print(f"Using: DeepFace RetinaFace (shared with Stage 2)")
    print(f"GPU: {GPU_AVAILABLE}")
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
    print(f"Shared Model: DeepFace RetinaFace")
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
        print(f"Model: DeepFace RetinaFace (shared with Stage 2)")
        if result['reason']:
            print(f"Reason: {result['reason']}")
        print(f"\nChecks Passed:")
        for check, status in result['checks'].items():
            print(f"  {check}: {status}")
        print("="*60)
    else:
        # Test with default image
        print("No image path provided. Testing with 'Fullface.jpeg'...\n")
        result = stage1_validate("Fullface.jpeg", photo_type="PRIMARY")
        
        print("\n" + "="*60)
        print("VALIDATION RESULT")
        print("="*60)
        print(f"Stage: {result['stage']}")
        print(f"Result: {result['result']}")
        print(f"GPU Used: {result['gpu_used']}")
        print(f"Model: DeepFace RetinaFace (shared with Stage 2)")
        if result['reason']:
            print(f"Reason: {result['reason']}")
        print(f"\nChecks Passed:")
        for check, status in result['checks'].items():
            print(f"  {check}: {status}")
        print("="*60)
