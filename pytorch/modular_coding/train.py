"""
Trains a PyTorch image classification model using device-agnostic code.

This is the main entry point that ties together every module in this folder:
data_setup -> model_builder -> engine -> utils.

Example:
    python train.py --num_epochs 5 --batch_size 32 --hidden_units 10 --learning_rate 0.001
"""
import argparse
import os
import zipfile
from pathlib import Path

import requests
import torch
from torchvision import transforms

import data_setup
import engine
import model_builder
import utils


def download_data(data_path: Path) -> Path:
    """Downloads the pizza_steak_sushi dataset if it isn't already present.

    Returns the path to the image folder containing train/ and test/.
    """
    image_path = data_path / "pizza_steak_sushi"

    if image_path.is_dir():
        print(f"[INFO] {image_path} already exists, skipping download.")
        return image_path

    print(f"[INFO] Did not find {image_path}, downloading it...")
    image_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "pizza_steak_sushi.zip"
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    with open(zip_path, "wb") as f:
        request = requests.get(url)
        print("[INFO] Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("[INFO] Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

    os.remove(zip_path)
    return image_path


def main():
    parser = argparse.ArgumentParser(description="Train a TinyVGG image classifier.")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of samples per batch.")
    parser.add_argument("--hidden_units", type=int, default=10,
                        help="Number of hidden units in the conv layers.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to a data dir containing train/ and test/ "
                             "subfolders. If omitted, the pizza_steak_sushi "
                             "dataset is downloaded automatically.")
    parser.add_argument("--model_name", type=str,
                        default="05_going_modular_tinyvgg_model.pth",
                        help="Filename for the saved model (.pt or .pth).")
    args = parser.parse_args()

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Setup directories
    if args.data_dir:
        image_path = Path(args.data_dir)
    else:
        image_path = download_data(Path("data"))
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Create transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Create DataLoaders and get class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        transform=data_transform,
        batch_size=args.batch_size,
    )

    # Create the model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names),
    ).to(device)

    # Set up loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.num_epochs,
        device=device,
    )

    # Save the model
    utils.save_model(
        model=model,
        target_dir="models",
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
