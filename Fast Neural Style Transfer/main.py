import argparse
import torch
import os
from training.trainer import Trainer
from inference.inference import Stylizer
from utils import load_image, save_image

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")

    # Mode Selection
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Mode: train or inference")

    # Training arguments
    parser.add_argument("--content_folder", type=str, default="data/content", help="Path to content images folder")
    parser.add_argument("--style_image", type=str, default="data/style.jpg", help="Path to style image")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")

    # Inference arguments
    parser.add_argument("--model_path", type=str, default="checkpoints/model_epoch_2.pth", help="Path to trained model for inference")
    parser.add_argument("--input_image", type=str, default="data/content/sample.jpg", help="Path to content image for stylization")
    parser.add_argument("--output_image", type=str, default="output/stylized.jpg", help="Path to save stylized image")

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.mode == "train":
        print("Starting training mode...")
        trainer = Trainer(
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            alpha=1e5,
            beta=1e10,
            content_folder=args.content_folder,
            style_path=args.style_image,
            device=device,
        )
        trainer.train(checkpoint_dir=args.checkpoint_dir)
        print("Training Completed")

    elif args.mode == "inference":
        print("Starting inference mode...")
        os.makedirs(os.path.dirname(args.output_image), exist_ok=True)
        stylizer = Stylizer(model_path=args.model_path, device=device)
        input_img = load_image(args.input_image, resize_value=1080)
        output_img = stylizer.stylize(input_img)
        save_image(args.output_image, output_img)
        print(f"Stylized image saved at {args.output_image}")

if __name__ == "__main__":
    main()
