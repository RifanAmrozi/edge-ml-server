from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
import tempfile
import urllib.request
import shutil


app = FastAPI(title="Shoplifting Detection API", version="1.0.0")

# Global variables
detector = None
OBJECT_MODEL_PATH = "yolo11s.pt"  # Model untuk object detection
POSE_MODEL_PATH = "yolo11n-pose.pt"  # Model untuk pose detection

@app.get("/")
async def root():
    return {"message": "Shoplifting Detection API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "detector_ready": detector is not None}

@app.on_event("startup")
async def startup_event():
    """Initialize models and detector on startup"""
    global detector
    
    try:
        print("Starting initialization...")
        
        # Set environment untuk headless server
        os.environ['DISPLAY'] = ''
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'  
        os.environ['MPLBACKEND'] = 'Agg'
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
        os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        
        # Import cv2 DI SINI, bukan di atas!
        try:
            import cv2
            print("OpenCV imported successfully")
        except ImportError as cv_error:
            print(f"OpenCV import failed: {cv_error}")
            # Try fallback opencv-python-headless
            try:
                import cv2
                print("OpenCV headless imported successfully")
            except ImportError:
                raise ImportError("Both opencv-python and opencv-python-headless failed to import")
        
        # Download OBJECT model if not exists
        if not os.path.exists(OBJECT_MODEL_PATH):
            print("Downloading object detection model...")
            # URL untuk yolo11s.pt (sesuai dengan konstruktor)
            object_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
            urllib.request.urlretrieve(object_url, OBJECT_MODEL_PATH)
            print("Object detection model downloaded successfully.")
        
        # Download POSE model if not exists
        if not os.path.exists(POSE_MODEL_PATH):
            print("Downloading pose model...")
            pose_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
            urllib.request.urlretrieve(pose_url, POSE_MODEL_PATH)
            print("Pose model downloaded successfully.")
        
        # Import and initialize detector with BOTH models
        from shoplifting_detection import PoseShopliftingDetector
        
        print("Attempting to initialize detector...")
        # ✅ Sesuai dengan konstruktor yang ada
        detector = PoseShopliftingDetector(
            model_path=OBJECT_MODEL_PATH,  # Parameter pertama: model_path untuk object detection
            pose_model=POSE_MODEL_PATH     # Parameter kedua: pose_model untuk pose detection
        )
        print("✅ Detector initialized successfully with both models.")
        print(f"   Object detection model: {OBJECT_MODEL_PATH}")
        print(f"   Pose detection model: {POSE_MODEL_PATH}")
        
    except ImportError as e:
        print(f"Import error during startup: {e}")
        print("OpenCV libraries not available. API will run in limited mode.")
    except Exception as e:
        print(f"Other error during startup: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

@app.post("/analyze_video/")
async def analyze_video(file: UploadFile = File(...)):
    # Check if detector is available
    if detector is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: Detector not initialized. This might be due to missing system libraries."
        )
    
    # Import cv2 di dalam fungsi, bukan global ✅
    try:
        import cv2
    except ImportError:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: OpenCV not available due to missing system libraries"
        )
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            _, suspicious_persons, critical_alerts = detector.process_frame(frame)

            # Process critical alerts
            for alert in critical_alerts:
                results.append({
                    "frame": frame_count,
                    "track_id": alert["track_id"],
                    "score": alert["score"],
                    "actions": alert["actions"],
                    "timestamp": alert["timestamp"],
                    "alert_type": "CRITICAL"
                })

        cap.release()
        
        return {
            "status": "success",
            "total_frames": frame_count,
            "alerts_count": len(results),
            "alerts": results,
            "models_used": {
                "object_model": OBJECT_MODEL_PATH,
                "pose_model": POSE_MODEL_PATH
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

@app.get("/startup_logs")
async def get_startup_logs():
    """Get detailed startup error logs"""
    global detector
    
    logs = {
        "detector_status": detector is not None,
        "models_downloaded": {
            "object_model": os.path.exists(OBJECT_MODEL_PATH),
            "pose_model": os.path.exists(POSE_MODEL_PATH)
        },
        "test_results": {}
    }
    
    # Test imports
    try:
        import cv2
        logs["test_results"]["opencv"] = {"success": True, "version": cv2.__version__}
    except Exception as e:
        logs["test_results"]["opencv"] = {"success": False, "error": str(e)}
    
    try:
        from ultralytics import YOLO
        logs["test_results"]["ultralytics"] = {"success": True}
        
        # Test loading models directly
        try:
            test_obj_model = YOLO(OBJECT_MODEL_PATH)
            logs["test_results"]["object_model_load"] = {"success": True}
        except Exception as e:
            logs["test_results"]["object_model_load"] = {"success": False, "error": str(e)}
            
        try:
            test_pose_model = YOLO(POSE_MODEL_PATH)
            logs["test_results"]["pose_model_load"] = {"success": True}
        except Exception as e:
            logs["test_results"]["pose_model_load"] = {"success": False, "error": str(e)}
            
    except Exception as e:
        logs["test_results"]["ultralytics"] = {"success": False, "error": str(e)}
    
    try:
        import mediapipe as mp
        logs["test_results"]["mediapipe"] = {"success": True}
    except Exception as e:
        logs["test_results"]["mediapipe"] = {"success": False, "error": str(e)}
    
    # Test detector import dan inisialisasi
    try:
        from shoplifting_detection import PoseShopliftingDetector
        logs["test_results"]["detector_import"] = {"success": True}
        
        # Test manual initialization
        try:
            test_detector = PoseShopliftingDetector(
                model_path=OBJECT_MODEL_PATH,
                pose_model=POSE_MODEL_PATH
            )
            logs["test_results"]["detector_init"] = {"success": True}
        except Exception as e:
            logs["test_results"]["detector_init"] = {"success": False, "error": str(e)}
            import traceback
            logs["test_results"]["detector_init"]["traceback"] = traceback.format_exc()
            
    except Exception as e:
        logs["test_results"]["detector_import"] = {"success": False, "error": str(e)}
    
    return logs

@app.get("/debug")
async def debug_startup():
    """Debug startup issues"""
    debug_info = {
        "models_exist": {
            "pose": os.path.exists(POSE_MODEL_PATH),
            "object": os.path.exists(OBJECT_MODEL_PATH)
        },
        "detector_status": detector is not None,
        "opencv_available": False,
        "ultralytics_available": False
    }
    
    # Test OpenCV
    try:
        import cv2
        debug_info["opencv_available"] = True
        debug_info["opencv_version"] = cv2.__version__
    except Exception as e:
        debug_info["opencv_error"] = str(e)
    
    # Test Ultralytics
    try:
        from ultralytics import YOLO
        debug_info["ultralytics_available"] = True
    except Exception as e:
        debug_info["ultralytics_error"] = str(e)
    
    # Test detector import
    try:
        from shoplifting_detection import PoseShopliftingDetector
        debug_info["detector_import"] = True
    except Exception as e:
        debug_info["detector_import_error"] = str(e)
    
    return debug_info

@app.get("/models_info")
async def get_models_info():
    """Get information about loaded models"""
    return {
        "pose_model": {
            "path": POSE_MODEL_PATH,
            "exists": os.path.exists(POSE_MODEL_PATH),
            "size": os.path.getsize(POSE_MODEL_PATH) if os.path.exists(POSE_MODEL_PATH) else 0
        },
        "object_model": {
            "path": OBJECT_MODEL_PATH,
            "exists": os.path.exists(OBJECT_MODEL_PATH),
            "size": os.path.getsize(OBJECT_MODEL_PATH) if os.path.exists(OBJECT_MODEL_PATH) else 0
        },
        "detector_ready": detector is not None
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)