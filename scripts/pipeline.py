import json
from pathlib import Path
 
import cv2
import matplotlib.pyplot as plt
 
from generate_rgb import ThermalToRGBGenerator
from yolo_detection import YOLODetector
from utils import list_images, summarize_detections
 
# -----------------------------
# Paths
# -----------------------------
THERMAL_DIR = "../input/thermal"
RGB_DIR = "../output/rgb"
DETECTION_DIR = "../output/detected"
JSON_DIR = "../output/json"
 
WEIGHTS_PATH = "../models/weights.pth"
 
 
# -----------------------------
# Display Images
# -----------------------------
def show_results(thermal_path, rgb_path, detected_path, object_count):
    thermal = cv2.imread(str(thermal_path))
    rgb = cv2.imread(str(rgb_path))
    detected = cv2.imread(str(detected_path))
 
    thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    detected = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
 
    plt.figure(figsize=(18, 6))
 
    plt.subplot(1, 3, 1)
    plt.imshow(thermal)
    plt.title("Thermal Image")
    plt.axis("off")
 
    plt.subplot(1, 3, 2)
    plt.imshow(rgb)
    plt.title("Generated RGB")
    plt.axis("off")
 
    plt.subplot(1, 3, 3)
    plt.imshow(detected)
    plt.title("YOLO Detection")
    plt.axis("off")
 
    plt.suptitle(
        "\n".join([f"{k}: {v}" for k, v in object_count.items()]),
        fontsize=11
    )
 
    plt.tight_layout()
    plt.show()
 
 
# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline():
 
    thermal_images = list_images(THERMAL_DIR)
 
    if not thermal_images:
        print(f"No thermal images found in {THERMAL_DIR}")
        return
 
    print("=" * 60)
    print("THERMAL OBJECT DETECTION PIPELINE")
    print("=" * 60)
 
    print(f"\nFound {len(thermal_images)} thermal image(s).\n")
 
    print("Loading Generator...")
    rgb_generator = ThermalToRGBGenerator(WEIGHTS_PATH)
 
    print("Loading YOLO...")
    detector = YOLODetector()
 
    print("\nStep 1 : Thermal → RGB")
    print("-" * 60)
 
    rgb_generator.generate_folder(
        THERMAL_DIR,
        RGB_DIR
    )
 
    print("\nStep 2 : RGB → Object Detection")
    print("-" * 60)
 
    detector.detect_folder(RGB_DIR)
 
    print("\nStep 3 : Showing Results")
    print("-" * 60)
 
    # Match each thermal image to its RGB/detected/JSON output by filename
    # (not by position in separately-sorted glob lists) - this stays
    # correct even if the output folders contain leftover files from
    # earlier runs, which was previously causing mismatched image pairs.
    for thermal in thermal_images:
 
        rgb = Path(RGB_DIR) / thermal.name
        detected = Path(DETECTION_DIR) / thermal.name
        json_file = Path(JSON_DIR) / f"{thermal.stem}.json"
 
        if not (rgb.exists() and detected.exists() and json_file.exists()):
            print(f"Skipping {thermal.name}: missing output file(s).")
            continue
 
        with open(json_file, "r") as f:
            data = json.load(f)
 
        print("\n" + "=" * 50)
        print(f"Image : {rgb.name}")
        print("=" * 50)
 
        print(f"Total Objects : {data['total_objects']}")
 
        print("\nObject Count")
 
        if len(data["object_count"]) == 0:
            print("No objects detected.")
 
        else:
            for obj, count in data["object_count"].items():
                print(f"{obj:<20}: {count}")
 
        show_results(
            thermal,
            rgb,
            detected,
            data["object_count"]
        )
 
    print("\nStep 4 : Overall Summary")
    print("-" * 60)
 
    summary = summarize_detections(JSON_DIR)
 
    print(json.dumps(summary, indent=4))
 
    print("\nPipeline Completed Successfully!")
 
 
if __name__ == "__main__":
    run_pipeline()
 