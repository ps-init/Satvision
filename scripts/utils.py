import json
from pathlib import Path

# Extensions we'll treat as valid input images
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    """
    Return a sorted list of image file paths found directly inside `folder`.
    """
    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize_detections(json_folder):
    """
    Read every detection JSON produced by YOLODetector.detect() in
    `json_folder` and combine them into one overall summary: how many
    images were processed, total objects found, and a combined count
    per class across the whole dataset.
    """
    json_folder = Path(json_folder)

    combined_counts = {}
    total_objects = 0
    image_count = 0

    for json_file in sorted(json_folder.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        image_count += 1
        total_objects += data.get("total_objects", 0)

        for class_name, count in data.get("object_count", {}).items():
            combined_counts[class_name] = combined_counts.get(class_name, 0) + count

    return {
        "images_processed": image_count,
        "total_objects_detected": total_objects,
        "object_count_by_class": combined_counts
    }