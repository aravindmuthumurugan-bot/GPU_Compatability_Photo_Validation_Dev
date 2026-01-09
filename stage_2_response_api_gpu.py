import os
import shutil
import json
import time
import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from stage_2_response_gpu import stage2_validate_optimized, configure_gpu

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Stage 2 Photo Validation API - GPU Optimized")

# Reduce workers since GPU operations are parallelized internally
MAX_WORKERS = 2  # Use fewer workers to avoid GPU memory conflicts


def convert_to_native_types(obj):
    """
    Recursively convert NumPy types to Python native types
    """
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


def process_single_image(image_path: str, profile_data: dict):
    """
    This function runs with GPU acceleration and early exit optimization.
    Configure GPU at the start of each worker process.
    """
    try:
        # Configure GPU for this worker
        configure_gpu()
        
        result = stage2_validate_optimized(
            image_path=image_path,
            profile_data=profile_data,
            existing_photos=[]
        )
        # Convert NumPy types to native Python types
        result = convert_to_native_types(result)
        
        return {
            "image": os.path.basename(image_path),
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "image": os.path.basename(image_path),
            "success": False,
            "error": str(e)
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
            "cuda_available": tf.test.is_built_with_cuda()
        })
    except Exception as e:
        return JSONResponse({
            "status": "degraded",
            "error": str(e)
        })


@app.post("/stage2/validate-images")
def validate_multiple_images(
    files: List[UploadFile] = File(...),
    profile_data: str = Form(...)
):
    """
    Upload multiple images and validate them with GPU acceleration
    """
    start_time = time.time()

    try:
        profile_data = json.loads(profile_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile_data JSON")

    if not files:
        raise HTTPException(status_code=400, detail="No images uploaded")

    saved_paths = []

    # Save files first
    for file in files:
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_paths.append(file_path)

    results = []

    # Process with limited workers to avoid GPU memory conflicts
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_image, path, profile_data): path
            for path in saved_paths
        }

        for future in as_completed(futures):
            results.append(future.result())

    # Cleanup uploaded files
    for path in saved_paths:
        try:
            os.remove(path)
        except:
            pass

    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    return JSONResponse({
        "total_images": len(files),
        "processed": len(results),
        "response_time_seconds": response_time,
        "results": results
    })


@app.post("/stage2/validate-single")
def validate_single_image(
    file: UploadFile = File(...),
    profile_data: str = Form(...)
):
    """
    Validate a single image with GPU acceleration
    """
    start_time = time.time()

    try:
        profile_data = json.loads(profile_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile_data JSON")

    # Save file
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process
    result = process_single_image(file_path, profile_data)

    # Cleanup
    try:
        os.remove(file_path)
    except:
        pass

    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    result["response_time_seconds"] = response_time

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    
    # Configure GPU before starting server
    print("\n" + "="*60)
    print("STARTING GPU-OPTIMIZED VALIDATION API")
    print("="*60)
    configure_gpu()
    
    uvicorn.run(
        "stage_2_response_api_gpu:app",
        host="0.0.0.0",
        port=8000,
        workers=1  # Use single worker to avoid GPU conflicts
    )
