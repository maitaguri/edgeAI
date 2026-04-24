import os
from PIL import Image
from glob import glob

dataset_path = "/Users/maitaguri/Documents/B3/Experiments/EdgeAI/dataset"
classes = ["ironed_white_shirt", "wrinkled_white_shirt"]

target_size = (224, 224)

for c in classes:
    class_dir = os.path.join(dataset_path, c)
    if not os.path.isdir(class_dir):
        print(f"Directory not found: {class_dir}")
        continue

    images = glob(os.path.join(class_dir, "*.[jp][pn]*[g]")) + glob(
        os.path.join(class_dir, "*.bmp")
    )
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                if img.size != target_size:
                    # Convert to RGB if needed
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    resized = img.resize(target_size, Image.LANCZOS)
                    resized.save(img_path)
                    print(f"Resized: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("Resize complete!")
