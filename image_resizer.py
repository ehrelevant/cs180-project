import cv2
import glob
import os

"""
REQUIREMENTS:
pip install opencv-python

HOW TO RUN:
python -m image_resizer

commands:
uv venv

.venv/Scripts/activate
"""


RAW_FOLDER_PATH = "potato_train/train"
OUTPUT_FOLDER_PATH = "potato_train/train_resized"
NEW_RESOLUTION = (512, 512)


def run():
    for folder in os.listdir(RAW_FOLDER_PATH):
        path = os.path.join(RAW_FOLDER_PATH, folder)
        output_path = os.path.join(OUTPUT_FOLDER_PATH, folder)

        try:
            os.mkdir(output_path)
            print(f"Directory '{output_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{output_path}' already exists.")

        jpeg_files = glob.glob(os.path.join(path, "*.jpeg"))
        for jpeg in jpeg_files:
            resize_image(jpeg, NEW_RESOLUTION)
    return


def resize_image(input_filepath: str, resolution: tuple[int, int]) -> None:
    # Load image
    image = cv2.imread(input_filepath)  # BGR format (OpenCV default)
    output_filepath = input_filepath.replace("/train", "/train_resized")

    # Resize to 512x512
    resized_image = cv2.resize(
        image, resolution, interpolation=cv2.INTER_AREA
    )  # INTER_AREA for downscaling

    # Save the resized image
    cv2.imwrite(output_filepath, resized_image)
    print(f"Saved to {output_filepath}")


if __name__ == "__main__":
    run()
