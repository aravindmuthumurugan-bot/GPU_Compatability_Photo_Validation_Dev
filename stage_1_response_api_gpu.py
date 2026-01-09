import os
import shutil
import time
import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from stage_1_response_gpu import stage1_validate, configure_gpu, GPU_AVAILABLE

UPLOAD_DIR = "uploads_stage1"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Stage-1 Image Sanity Check (GPU Optimized)",
    version="2.0"
)

# Use fewer workers for Stage 1 (lighter processing than Stage 2)
MAX_WORKERS = 4


def convert_to_native_types(obj):
    """Convert NumPy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    return obj


def process_single_image(image_path: str, photo_type: str, person_id: str, filename: str):
    """Process a single image with GPU acceleration"""
    try:
        # Configure GPU for this worker
        configure_gpu()
        
        result = stage1_validate(
            image_path=image_path,
            photo_type=photo_type
        )
        
        # Add metadata
        result.update({
            "person_id": person_id,
            "file_name": filename
        })
        
        # Convert NumPy types
        result = convert_to_native_types(result)
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": {
                "person_id": person_id,
                "file_name": filename,
                "stage": 1,
                "result": "REJECT",
                "reason": f"Processing error: {str(e)}",
                "checks": {},
                "gpu_used": GPU_AVAILABLE
            }
        }


@app.get("/")
def root():
    """API root endpoint"""
    return {
        "service": "Stage 1 Photo Validation",
        "version": "2.0",
        "gpu_enabled": GPU_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "validate_single": "/stage1/validate/single",
            "validate_batch": "/stage1/validate/batch"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint with GPU status"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        return JSONResponse({
            "status": "healthy",
            "tensorflow_version": tf.__version__,
            "gpus_available": len(gpus),
            "gpu_names": [str(gpu) for gpu in gpus],
            "cuda_available": tf.test.is_built_with_cuda(),
            "stage": 1
        })
    except Exception as e:
        return JSONResponse({
            "status": "degraded",
            "error": str(e),
            "stage": 1
        })


@app.post("/stage1/validate/single")
async def stage1_validate_single(
    file: UploadFile = File(...),
    person_id: str = Form(...),
    photo_type: str = Form("PRIMARY")
):
    """
    Stage-1 validation for a single image with GPU acceleration
    """
    start_time = time.time()

    if not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id is required")

    if photo_type not in {"PRIMARY", "SECONDARY"}:
        raise HTTPException(
            status_code=400,
            detail="photo_type must be PRIMARY or SECONDARY"
        )

    # Save file
    temp_filename = file.filename
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process
        result_data = process_single_image(temp_path, photo_type, person_id, file.filename)

        end_time = time.time()
        response_time = round(end_time - start_time, 3)

        result_data["result"]["response_time_seconds"] = response_time

        return JSONResponse(result_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/stage1/validate/batch")
async def stage1_validate_batch_api(
    files: List[UploadFile] = File(...),
    person_id: str = Form(...),
    photo_type: str = Form("PRIMARY")
):
    """
    Stage-1 batch validation for multiple images with GPU acceleration
    """
    start_time = time.time()

    if not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id is required")

    if photo_type not in {"PRIMARY", "SECONDARY"}:
        raise HTTPException(
            status_code=400,
            detail="photo_type must be PRIMARY or SECONDARY"
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one image must be uploaded"
        )

    # Save all files first
    saved_paths = []
    filenames = []

    for file in files:
        temp_filename = file.filename
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_paths.append(temp_path)
        filenames.append(file.filename)

    results = []

    # Process in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_image, path, photo_type, person_id, filename): (path, filename)
            for path, filename in zip(saved_paths, filenames)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result["result"])

    # Cleanup
    for path in saved_paths:
        try:
            os.remove(path)
        except:
            pass

    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    # Calculate summary
    passed = sum(1 for r in results if r.get("result") == "PASS")
    rejected = len(results) - passed

    return JSONResponse({
        "person_id": person_id,
        "photo_type": photo_type,
        "total_images": len(files),
        "passed": passed,
        "rejected": rejected,
        "response_time_seconds": response_time,
        "gpu_used": GPU_AVAILABLE,
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("STARTING STAGE 1 GPU-OPTIMIZED VALIDATION API")
    print("="*60)
    configure_gpu()
    
    uvicorn.run(
        "stage_1_api_gpu:app",
        host="0.0.0.0",
        port=8001,  # Different port from Stage 2
        workers=1   # Single worker for GPU
    )
