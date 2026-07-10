import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# Extensions we'll treat as valid input images
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class ThermalToRGBGenerator:
    def __init__(self, weights_path="../models/weights.pth"):
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

        # Load trained weights
        self.generator.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )

        self.generator.eval()

        # Same preprocessing used during training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
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

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.generator(tensor)

        output = output.squeeze(0).cpu()
        rgb = transforms.ToPILImage()(output)

        return rgb

    def generate(self, input_image, output_image=None):
        """
        Run inference on a single image.
        """

        image = Image.open(input_image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.generator(tensor)

        output = output.squeeze(0).cpu()

        rgb = transforms.ToPILImage()(output)

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
        "../models/weights.pth"
    )

    model.generate_folder(
        "../input/thermal",
        "../output/rgb"
    )
