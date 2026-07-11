import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from torchvision.transforms import v2

import segmentation_models_pytorch as smp

# Extensions we'll treat as valid input images
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def apply_clahe(pil_img):
    """
    Matches the exact CLAHE preprocessing used in training.
    Boosts faint thermal features before passing to the model.
    This step is critical for achieving good results - without it,
    the model receives underprocessed thermal data and produces poor RGB conversions.
    """
    img_np = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_3ch)


class ThermalToRGBGenerator:
    def __init__(self, weights_path="models/weights.pth"):
        """
        Load the trained Generator.
        """

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        # Create Generator (same architecture used during training)
        self.generator = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation="sigmoid",
            decoder_attention_type="scse"
        ).to(self.device)

        # Load trained weights (handles both a raw state_dict and a
        # full training checkpoint that wraps it in 'generator_state_dict')
        loaded_data = torch.load(weights_path, map_location=self.device)

        if isinstance(loaded_data, dict) and "generator_state_dict" in loaded_data:
            state_dict = loaded_data["generator_state_dict"]
        else:
            state_dict = loaded_data

        self.generator.load_state_dict(state_dict, strict=True)
        self.generator.eval()

        # Exact same transform pipeline used in training/inference (v2 API)
        self.transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def generate_array(self, input_array):
        """
        Run inference on a single image in memory.
        
        Args:
            input_array: numpy array or PIL Image
            
        Returns:
            PIL Image of the generated RGB output
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(input_array, np.ndarray):
            image = Image.fromarray(input_array.astype('uint8')).convert("RGB")
        else:
            image = input_array.convert("RGB")

        # Apply CLAHE before anything else - matches training pipeline
        # THIS IS CRITICAL FOR GOOD RESULTS
        image = apply_clahe(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.generator(tensor)

        output = output.squeeze(0).cpu()
        rgb = v2.ToPILImage()(output)

        return rgb

    def generate(self, input_image, output_image=None):
        """
        Run inference on a single image, matching the training-time preprocessing exactly (CLAHE + v2 transforms).
        """

        raw_img = Image.open(input_image).convert("RGB")

        # Apply CLAHE before anything else - matches training pipeline
        t_img_clahe = apply_clahe(raw_img)

        tensor = self.transform(t_img_clahe).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.generator(tensor)

        # Sigmoid activation guarantees output is in [0, 1]
        output = output.squeeze(0).cpu().clamp(0, 1)

        rgb = Image.fromarray(
            (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        if output_image is not None:

            Path(os.path.dirname(output_image)).mkdir(
                parents=True,
                exist_ok=True
            )

            rgb.save(output_image)

            print(f"Saved RGB image to {output_image}")

        return rgb

    def generate_folder(self, input_dir, output_dir):
        """
        Run inference on every image found in input_dir, saving each
        result into output_dir with the same filename.
        """

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        image_paths = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            print(f"No images found in {input_dir.resolve()}")
            return

        print(f"Found {len(image_paths)} image(s) in {input_dir}")

        for image_path in image_paths:
            output_path = output_dir / image_path.name
            print(f"Processing {image_path.name}...")
            self.generate(str(image_path), str(output_path))


if __name__ == "__main__":

    model = ThermalToRGBGenerator(
        "models/weights.pth"
    )

    model.generate_folder(
        "input/thermal",
        "output/rgb"
    )
