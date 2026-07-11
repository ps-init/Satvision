# Satvision

**Thermal Infrared to RGB Image Translation & Critical Infrastructure Detection**

Satvision converts thermal infrared imagery into RGB images using a Pix2Pix GAN and automatically detects critical infrastructure (vehicles, buildings, roads) using YOLOv11.

## Features

- 🌡️ **Thermal-to-RGB Translation**: Converts thermal infrared images to realistic RGB using a U-Net based Pix2Pix generator
- 🎯 **Object Detection**: Detects vehicles, buildings, and roads with YOLOv11
- 🚀 **Dual Interface**: FastAPI REST backend + Flask web UI
- ⚡ **Real-time Processing**: Process images on-demand or batch process entire folders
- 📊 **Detailed Analytics**: Per-image detection counts, confidence scores, and bounding boxes

## Project Structure

```
Satvision/
├── api.py                    # FastAPI server (main backend)
├── app.py                    # Flask web UI server
├── inference.py              # Standalone inference script (Colab-compatible)
├── scripts/
│   ├── generate_rgb.py       # Thermal-to-RGB generator with CLAHE preprocessing
│   ├── yolo_detection.py     # YOLO object detector
│   ├── utils.py              # Utility functions
│   └── pipeline.py           # Batch processing pipeline
├── models/
│   └── weights.pth           # Pre-trained Pix2Pix generator weights
├── templates/
│   └── index.html            # Web UI frontend
├── static/
│   ├── js/
│   │   └── script.js         # Frontend logic
│   ├── css/
│   │   └── style.css         # Styling
│   └── images/               # UI placeholders
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/ps-init/Satvision.git
cd Satvision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: FastAPI Backend Only

```bash
python api.py
# API runs at http://localhost:8000
# Docs available at http://localhost:8000/docs
```

**Upload and process an image:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@thermal_image.png"
```

**Response:**
```json
{
  "filename": "thermal_image.png",
  "total_objects": 5,
  "object_count": {
    "car": 3,
    "building": 2
  },
  "detections": [
    {
      "class": "car",
      "confidence": 0.92,
      "bbox": [100, 150, 250, 300]
    }
  ],
  "annotated_image_base64": "data:image/png;base64,..."
}
```

### Option 2: Flask Web UI + FastAPI Backend

**Terminal 1 - Start FastAPI:**
```bash
python api.py
# Runs on http://localhost:8000
```

**Terminal 2 - Start Flask:**
```bash
python app.py
# Runs on http://localhost:5000
```

Then open http://localhost:5000 in your browser.

### Option 3: Batch Processing Pipeline

```bash
python scripts/pipeline.py
```

This script:
1. Reads all thermal images from `../input/thermal/`
2. Generates RGB images → `../output/rgb/`
3. Detects objects → `../output/detected/` + `../output/json/`
4. Displays side-by-side comparisons

## Critical: CLAHE Preprocessing

**The model REQUIRES CLAHE (Contrast-Limited Adaptive Histogram Equalization) preprocessing on thermal images.** Without this step, the RGB generation produces poor results.

All methods (`generate_array()`, `generate()`, `generate_folder()`) automatically apply CLAHE internally. This was the critical fix that improved real-time API results to match batch processing performance.

**CLAHE Parameters:**
- `clipLimit=2.0` - Contrast boost strength
- `tileGridSize=(8, 8)` - Local region size

## Model Architecture

### Thermal-to-RGB Generator
- **Architecture**: U-Net with ResNet34 encoder
- **Encoder Weights**: ImageNet pre-trained
- **Activation**: Sigmoid (outputs in [0, 1])
- **Decoder Attention**: SCSE (Spatial and Channel Squeeze-and-Excitation)

### Object Detector
- **Architecture**: YOLOv11 Nano (lightweight)
- **Classes**: vehicle, building, road
- **Weights**: Auto-downloaded from Ultralytics on first run

## API Endpoints

### POST `/detect`
Upload a thermal image for processing.

**Request:**
```
Content-Type: multipart/form-data
file: <thermal_image>
```

**Response:**
```json
{
  "filename": "string",
  "total_objects": 0,
  "object_count": { "class_name": count },
  "detections": [ { "class": "string", "confidence": float, "bbox": [x1, y1, x2, y2] } ],
  "annotated_image_base64": "string"
}
```

### GET `/health`
Check API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Performance Notes

- **GPU**: ~200-500ms per image
- **CPU**: ~2-5s per image
- **Model Size**: ~98MB (weights.pth)
- **Batch Size**: 1 (configurable in code)

## Troubleshooting

### "Models not loaded" error
- Ensure `models/weights.pth` exists
- Check file permissions and disk space

### Poor RGB generation quality
- Verify CLAHE is being applied (see logs for confirmation)
- Check that thermal image has sufficient contrast

### CORS errors in web UI
- FastAPI CORS middleware is enabled by default
- Modify `allow_origins` in `api.py` if needed

### Out of memory errors
- Run on GPU instead of CPU
- Reduce batch size (currently 1)
- Process smaller images

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "api.py"]
```

### Environment Variables
```bash
export WEIGHTS_PATH="models/weights.pth"
export FLASK_PORT=5000
export API_PORT=8000
```

## Files Reference

| File | Purpose |
|------|---------|
| `api.py` | FastAPI backend serving `/detect` and `/health` endpoints |
| `app.py` | Flask server for web UI at `/` |
| `inference.py` | Standalone inference script (Colab-compatible) |
| `scripts/generate_rgb.py` | U-Net generator with CLAHE preprocessing |
| `scripts/yolo_detection.py` | YOLO wrapper for object detection |
| `scripts/utils.py` | Utility functions for batch processing |
| `scripts/pipeline.py` | Full batch processing pipeline orchestrator |
| `models/weights.pth` | Trained Pix2Pix generator weights (98MB) |
| `requirements.txt` | Python package dependencies |

## License

MIT License - See LICENSE file for details.

## Contributing

Pull requests welcome! Please ensure:
1. CLAHE preprocessing is applied to any new thermal processing code
2. All functions include docstrings
3. Error handling includes meaningful messages

## Support

For issues, questions, or contributions, please open a GitHub issue.
