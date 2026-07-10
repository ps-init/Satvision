from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from scripts.generate_rgb import ThermalToRGBGenerator
from scripts.yolo_detection import YOLODetector

# Initialize FastAPI app
app = FastAPI(title="Thermal Object Detection API", version="1.0.0")

# Global model instances (loaded once at startup)
rgb_generator = None
yolo_detector = None


@app.on_event("startup")
async def load_models():
    """
    Load models on API startup for reuse across requests.
    """
    global rgb_generator, yolo_detector
    
    print("Loading Thermal-to-RGB Generator...")
    rgb_generator = ThermalToRGBGenerator(weights_path="models/weights.pth")
    
    print("Loading YOLO Detector...")
    yolo_detector = YOLODetector(model_name="yolo11n.pt")
    
    print("Models loaded successfully!")


@app.post("/detect")
async def detect_thermal_image(file: UploadFile = File(...)):
    """
    Full pipeline: Thermal Image → RGB Generation → Object Detection
    
    Args:
        file: Uploaded thermal image (PNG, JPG, etc.)
        
    Returns:
        JSON response with detections and optional base64-encoded annotated image
    """
    try:
        # Validate file upload
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read uploaded image
        contents = await file.read()
        thermal_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Step 1: Generate RGB from thermal
        print(f"Processing {file.filename}...")
        rgb_image = rgb_generator.generate_array(thermal_image)
        
        # Step 2: Run YOLO detection on RGB
        detection_result = yolo_detector.detect_array(rgb_image)
        
        # Step 3: Prepare response
        response_data = {
            "filename": file.filename,
            "total_objects": detection_result["total_objects"],
            "object_count": detection_result["object_count"],
            "detections": detection_result["detections"]
        }
        
        # Optional: Include base64-encoded annotated image for visualization
        annotated_img_bytes = io.BytesIO()
        detection_result["annotated_image"].save(annotated_img_bytes, format="PNG")
        annotated_img_b64 = base64.b64encode(annotated_img_bytes.getvalue()).decode("utf-8")
        
        response_data["annotated_image_base64"] = annotated_img_b64
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running and models are loaded.
    """
    return {
        "status": "healthy",
        "models_loaded": rgb_generator is not None and yolo_detector is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
