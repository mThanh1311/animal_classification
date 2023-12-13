import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from src.model import AnimalModel

def get_args():
    parser = argparse.ArgumentParser(description="Animal Classifier")
    parser.add_argument("-s", "--size", type=int, default=224)
    parser.add_argument("-i", "--input_path", type=str, default="test_videos/test_video_input.mp4")
    parser.add_argument("-o", "--output_path", type=str, default="test_videos/test_video_output.mp4")
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

    if not args.input_path:
        print("An image must be provided")
        exit(0)


    cap = cv2.VideoCapture(args.input_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"MP4V"), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
    counter = 0
    while cap.isOpened():
        print(counter)
        counter += 1
        flag, frame = cap.read()
        if not flag:
            break
        image = cv2.resize(frame, (args.size, args.size))
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        # image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device).float()[None, :, :, :]
        softmax = nn.Softmax()
        with torch.no_grad():
            prediction = model(image)
        probs = softmax(prediction)
        max_value, max_index = torch.max(probs, dim=1)
        confidence_str = str(max_value.item())
        cv2.putText(frame, f"{categories[max_index]}: {confidence_str}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()

if __name__ == '__main__':
    args = get_args()
    test(args)