import argparse
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(description="SCBDT Signature Verification")
    parser.add_argument("--model", type=str, default="HDSV-Net", 
                        choices=["HDSV-Net", "Siamese-CNN", "ResNet50", "MobileNetV2", "ViT"],
                        help="පුහුණු කළ යුතු ආකෘතිය තෝරන්න")
    parser.add_argument("--data_dir", type=str, default="data/", help="දත්ත ඇති ෆෝල්ඩරය")
    
    args = parser.parse_args()
    print(f"🚀 {args.model} පුහුණු කිරීම ආරම්භ කරමින්...")
    train_model(args.model, args.data_dir)

if __name__ == "__main__":
    main()