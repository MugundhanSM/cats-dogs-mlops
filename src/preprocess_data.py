import os
import random
from PIL import Image

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

MAX_IMAGES_PER_CLASS = 2500
IMAGE_SIZE = (128, 128)
SPLIT_RATIO = (0.8, 0.1, 0.1)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def resize_and_crop(img):
    """Resize while keeping aspect ratio, then center crop."""
    img.thumbnail((150, 150))  # keep ratio
    width, height = img.size

    left = (width - IMAGE_SIZE[0]) // 2
    top = (height - IMAGE_SIZE[1]) // 2
    right = left + IMAGE_SIZE[0]
    bottom = top + IMAGE_SIZE[1]

    return img.crop((left, top, right, bottom))


def process_and_save(image_path, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = resize_and_crop(img)
        img.save(save_path)
    except Exception:
        print(f"Skipping bad image: {image_path}")


def preprocess():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for category in ["cat", "dog"]:
        input_path = os.path.join(INPUT_DIR, category)

        if not os.path.exists(input_path):
            raise Exception(f"Missing folder: {input_path}")

        files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(files)
        files = files[:MAX_IMAGES_PER_CLASS]

        total = len(files)
        train_end = int(total * SPLIT_RATIO[0])
        val_end = train_end + int(total * SPLIT_RATIO[1])

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split_name, split_files in splits.items():

            split_output_dir = os.path.join(
                OUTPUT_DIR, split_name, category
            )
            os.makedirs(split_output_dir, exist_ok=True)

            for file in split_files:
                img_path = os.path.join(input_path, file)
                out_path = os.path.join(split_output_dir, file)

                process_and_save(img_path, out_path)

        print(f"{category}: {total} images processed")


if __name__ == "__main__":
    preprocess()
