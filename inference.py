import os
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision.transforms import v2

def apply_clahe(pil_img):
    """
    Matches the exact CLAHE preprocessing used in training.
    Boosts faint thermal features before passing to the model.
    """
    img_np = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_3ch)

# --- CONFIGURATION (Edit these paths in Colab) ---
INPUT_IMAGE_PATH = "/content/test_image.jpg"
WEIGHTS_PATH = "/content/weights.pth"
# -----------------------------------------------

def run_inference():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input file '{INPUT_IMAGE_PATH}' not found.")
        return
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file '{WEIGHTS_PATH}' not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Generator (Exact same architecture as iris.py)
    print("Loading Generator...")
    generator = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation="sigmoid",
        decoder_attention_type="scse"
    ).to(device)

    # Load trained weights safely
    loaded_data = torch.load(WEIGHTS_PATH, map_location=device)

    # If the user points to the checkpoint file instead of weights.pth, extract the generator state dict
    if 'generator_state_dict' in loaded_data:
        state_dict = loaded_data['generator_state_dict']
    else:
        state_dict = loaded_data

    # strict=True guarantees that if there is a mismatch, it crashes loudly instead of silently failing
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    print("Weights loaded successfully.")

    # 2. Preprocess the Input Image
    print(f"Processing input image: {INPUT_IMAGE_PATH}")
    try:
        raw_img = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    # Apply the exact same CLAHE logic from training
    t_img_clahe = apply_clahe(raw_img)

    # Apply the exact same v2 transforms from training
    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Transform outputs a tensor of shape [3, 256, 256]. We unsqueeze to add batch dimension [1, 3, 256, 256]
    input_tensor = transform(t_img_clahe).unsqueeze(0).to(device)

    # 3. Generate Prediction
    print("Generating RGB prediction...")
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # 4. Post-process and Save
    # The generator has a sigmoid activation, so outputs are strictly in [0, 1]
    pred_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)

    # Let's save a side-by-side comparison (Input vs Output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Thermal (we show the CLAHE version to see what the model actually saw)
    t_np = input_tensor.squeeze(0).float().cpu().permute(1, 2, 0).numpy()
    t_np = np.clip(t_np, 0.0, 1.0)

    axes[0].imshow(t_np)
    axes[0].set_title("Input (Thermal + CLAHE)")
    axes[0].axis('off')

    axes[1].imshow(pred_np)
    axes[1].set_title("Generator Prediction (RGB)")
    axes[1].axis('off')


    plt.tight_layout()
    # In Colab, plt.show() displays the image directly beneath the cell
    plt.show()
    print(f"Success! Inference complete.")

if __name__ == "__main__":
    run_inference()
