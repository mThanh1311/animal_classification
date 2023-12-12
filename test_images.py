import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from src.model import AnimalModel


def get_args():
    parser = argparse.ArgumentParser(description="Animal Classifier")
    parser.add_argument("-s", "--size", type=int, default=224)
    parser.add_argument('-i', '--image_path', type=str, default="test_images/")
    parser.add_argument('-c', '--checkpoint_path', type=str, default='checkpoint/best.pt')
    args = parser.parse_args()
    return args


def test(args):
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AnimalModel(num_classes=len(categories)).to(device)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("A checkpoint must be provided")
        exit(0)

    if not args.image_path:
        print("An image must be provided")
        exit(0)

    images_path = [os.path.join(args.image_path, img_name) for img_name in os.listdir(args.image_path) if img_name.endswith('.jpg')]
    num_cols = 4
    num_rows = (len(images_path) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

    for ax in axes.flatten():
        ax.axis('off')

    for i, image_path in enumerate(images_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.size, args.size))
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device).float()

        softmax = nn.Softmax()
        with torch.no_grad():
            prediction = model(image)
        probs = softmax(prediction)
        max_value, max_index = torch.max(probs, dim=1)

        row, col = divmod(i, num_cols)

        axes[row, col].imshow(image.squeeze().permute(1, 2, 0))
        axes[row, col].set_title("{}:{:.4}".format(categories[max_index], max_value[0]))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = get_args()
    test(args)