import json
from pathlib import Path

import cv2
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_name="yolo11n.pt"):
        """
        Load YOLO model.
        """

        self.model = YOLO(model_name)

        print("YOLO model loaded successfully!")

    def detect(self, image_path,
               output_image="../output/detected",
               output_json="../output/json"):

        image_path = Path(image_path)

        output_image = Path(output_image)
        output_json = Path(output_json)

        output_image.mkdir(parents=True, exist_ok=True)
        output_json.mkdir(parents=True, exist_ok=True)

        results = self.model(str(image_path), verbose=False)

        result = results[0]

        annotated = result.plot()

        detected_image = output_image / image_path.name

        cv2.imwrite(str(detected_image), annotated)

        detections = []

        object_count = {}

        for box in result.boxes:

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self.model.names[cls]

            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

            object_count[class_name] = object_count.get(class_name, 0) + 1

        json_data = {
            "image": image_path.name,
            "total_objects": len(detections),
            "object_count": object_count,
            "detections": detections
        }

        json_file = output_json / (image_path.stem + ".json")

        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"Processed: {image_path.name}")
        print(f"\nImage: {image_path.name}")
        print("-" * 35)

        for name, count in object_count.items():
            print(f"{name:15} : {count}")

        print("-" * 35)
        print(f"Total Objects : {len(detections)}\n")

        return json_data

    def detect_folder(self,
                      input_folder="../output/rgb"):

        input_folder = Path(input_folder)

        images = sorted([
            p for p in input_folder.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ])

        if not images:
            print("No RGB images found.")
            return

        print(f"Found {len(images)} RGB image(s)\n")

        for img in images:
            self.detect(img)


if __name__ == "__main__":

    detector = YOLODetector()

    detector.detect_folder()