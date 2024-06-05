"""

    Wenqing Fan
    Hang Yin
    cs5330 24sp final project

    We imported some pokemon card pictures and find the ROI
    to see if we can classify the card into right version set

"""


import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

from p5_task1abcd import Net
    
# custom Image dataset to make the iterator to load the data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [name for name in os.listdir(
            img_dir) if name.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        original_image = image
        if self.transform:
            image = self.transform(image)
        return image, original_image

# To plot the predictions for new inputs
def show_pred_inputs(custom_dataset, network):
    # Make predictions for each image
    fig = plt.figure()
    for i in range(16):
        image, original_image = custom_dataset[i]
        # print(f"image is :{image}")

        with torch.no_grad():
            output = network(image)
            print(f"output{i} is : {output}")
            prediction = output.data.max(1, keepdim=True)[1]

        plt.subplot(4, 4, i+1)

        plt.imshow(original_image, cmap='gray')
        plt.title(f"Prediction: {prediction[0][0]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# main function to load the state of the network
# and show the predicted for the hand written digits
def main(argv):
    # Load network model
    network = Net()
    network.fc2 = nn.Linear(50, 2)
    network.load_state_dict(torch.load('./results/model.pth'))
    network.eval()

    # Define the location of images
    data_dir = './data/cards'

    # Define a custom function to crop the ROI
    def crop_roi(img):
        # Assuming img is a PIL Image
        # Crop the left bottom region of size
        cropped_img = img.crop((40, img.height - 65, 80, img.height - 25))
        return cropped_img
    
    # Define a custom transform function to invert images
    def invert_image(image):
        # Convert PIL Image to numpy array
        np_image = np.array(image)

        # Invert the image
        inverted_image = 255 - np_image

        # Convert numpy array back to PIL Image
        inverted_image = Image.fromarray(inverted_image)

        return inverted_image
    
    # Define transformations to apply to the images
    transform = transforms.Compose([
        transforms.Lambda(crop_roi),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.Lambda(invert_image),
        transforms.ToTensor()
    ])

    # Load a custom dataset
    custom_dataset = CustomImageDataset(
        img_dir=data_dir,
        transform=transform
    )

    show_pred_inputs(custom_dataset, network)

    img_path = "test_pokemon.png"
    test_pokemon_image = Image.open(img_path)
    test_pokemon_image_transform = transform(test_pokemon_image)
    with torch.no_grad():
        new_output = network(test_pokemon_image_transform)
        print(f"output is : {new_output}")
        new_prediction = new_output.data.max(1, keepdim=True)[1]

    plt.subplot(1, 2, 1)

    plt.imshow(test_pokemon_image_transform[0], cmap='gray')
    plt.subplot(1, 2, 2)

    plt.imshow(test_pokemon_image, cmap='gray')
    plt.title(f"Prediction: {new_prediction[0][0]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    import sys
    main(sys.argv)
