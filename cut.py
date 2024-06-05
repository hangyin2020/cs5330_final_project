import os

from PIL import Image

def cut_and_save_image(input_image_path, output_image_path, left_corner=(42, 857), shot_size=(35, 35)):
    # Open the input image
    input_image = Image.open(input_image_path)

    # Define the box region to crop (left, upper, right, lower)
    box = (left_corner[0], left_corner[1], left_corner[0] + shot_size[0], left_corner[1] + shot_size[1])

    # Crop the image
    cropped_image = input_image.crop(box)

    # Save the cropped image
    cropped_image.save(output_image_path)
    print("Cropped image saved successfully.")

def main():
    # Define paths
    input_directory = "./data/cards"
    output_directory = "./data/symbol_train/crown_and_zenith"

    for filename in os.listdir(input_directory):
        # Check if the file is an image
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_directory, filename)
            output_image_path = os.path.join(output_directory, f"cropped_{filename}")
            cut_and_save_image(input_image_path, output_image_path)

if __name__ == "__main__":
    main()