"""
Loads a trained TinyVGG model and makes a prediction on a single image.

Example:
    python predictions.py --image data/pizza_steak_sushi/test/pizza/some_image.jpg \
        --model_path models/05_going_modular_tinyvgg_model.pth
"""
import argparse
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

import model_builder


def predict_on_image(
    image_path: str,
    model_path: str,
    class_names: list,
    hidden_units: int = 10,
    image_size: tuple = (64, 64),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Loads a saved model and predicts the class of a single image."""
    # Rebuild the model architecture and load the saved weights
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=hidden_units,
        output_shape=len(class_names),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load and preprocess the image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.0
    transform = transforms.Resize(image_size)
    target_image = transform(target_image).unsqueeze(dim=0).to(device)

    # Predict
    with torch.inference_mode():
        pred_logits = model(target_image)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)

    predicted_class = class_names[pred_label.item()]
    confidence = pred_probs.max().item()
    print(f"[INFO] Image: {image_path}")
    print(f"[INFO] Predicted class: {predicted_class} | Confidence: {confidence:.3f}")
    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the target image.")
    parser.add_argument("--model_path", type=str,
                        default="models/05_going_modular_tinyvgg_model.pth",
                        help="Path to the saved model state_dict.")
    parser.add_argument("--hidden_units", type=int, default=10,
                        help="Hidden units the model was trained with "
                             "(must match training).")
    parser.add_argument("--class_names", type=str, nargs="+",
                        default=["pizza", "steak", "sushi"],
                        help="Ordered list of class names.")
    args = parser.parse_args()

    predict_on_image(
        image_path=args.image,
        model_path=args.model_path,
        class_names=args.class_names,
        hidden_units=args.hidden_units,
    )


if __name__ == "__main__":
    main()
