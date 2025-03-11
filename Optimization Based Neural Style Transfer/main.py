import argparse
from style_transfer import NeuralStyleTransfer
from visualize_content import visualize_content_rep
from visualize_style import visualize_style_rep

def main():
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer: Run full style transfer or visualize content/style representations."
    )
    parser.add_argument(
        "--content",
        type=str,
        help="Path to the content image."
    )
    parser.add_argument(
        "--style",
        type=str,
        help="Path to the style image."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["transfer", "content", "style", "all"],
        default="all",
        help="Mode to run: 'transfer' for full style transfer, 'content' for visualizing content representation, 'style' for visualizing style representation, or 'all' for running all."
    )
    args = parser.parse_args()

    if args.mode in ["transfer", "all"]:
        if not args.content or not args.style:
            parser.error("For 'transfer' or 'all' mode, both --content and --style image paths are required.")
    elif args.mode == "content":
        if not args.content:
            parser.error("For 'content' mode, the --content image path is required.")
        if not args.style:
            args.style = args.content
    elif args.mode == "style":
        if not args.style:
            parser.error("For 'style' mode, the --style image path is required.")
        if not args.content:
            args.content = args.style

    # Instantiate NeuralStyleTransfer with the available paths
    nst = NeuralStyleTransfer(args.content, args.style)

    if args.mode in ["transfer", "all"]:
        print("Running full style transfer...")
        generated_image = nst.style_transfer()
        generated_image.show()  

    if args.mode in ["content", "all"]:
        print("Visualizing content representation...")
        content_vis = visualize_content_rep(nst)
        content_vis.show()  

    if args.mode in ["style", "all"]:
        print("Visualizing style representation...")
        style_vis = visualize_style_rep(nst)
        style_vis.show()  
        
if __name__ == "__main__":
    main()
