import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.dataset import Animal
from src.model import AnimalModel
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
from tqdm import tqdm

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description="Animal Classifier")
    parser.add_argument('-p', '--data_path', type=str, default="../Dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-o', '--optimizer', type=str, choices=["SGD", "Adam"], default="Adam")
    parser.add_argument('-l', '--lr', type=float, default=0.001)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-t', '--tensorboard_path', type=str, default='tensorboard')
    parser.add_argument('-a', '--trained_path', type=str, default='checkpoint')
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])

    train_set = Animal(root = "./Dataset", train=True, transform=transform)
    valid_set = Animal(root = "./Dataset", train=False, transform=transform)

    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True
    }

    valid_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": True
    }

    train_loader = DataLoader(train_set, **train_params)
    valid_loader = DataLoader(valid_set, **valid_params)

    model = AnimalModel(num_classes=len(train_set.categories)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)

    if not os.path.isdir(args.trained_path):
        os.mkdir(args.trained_path)

    writer = SummaryWriter(args.tensorboard_path)
    num_iters = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        #TRAIN
        model.train()
        losses = []
        progress_bar = tqdm(train_loader, colour="white")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            #Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            progress_bar.set_description("Epoch {}/{}.  Loss value: {:.4}".format(epoch + 1, args.epochs, loss_value))
            losses.append(loss_value)
            writer.add_scalar("Train/Loss", np.mean(losses), epoch*num_iters+iter)

        #VALID
        model.eval()
        losses = []
        all_predictions = []
        all_gts = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                #Forward pass
                predictions_valid = model(images)
                max_idx = torch.argmax(predictions_valid, 1)

                loss = criterion(predictions_valid, labels)
                losses.append(loss.item())
                all_gts.extend(labels.tolist())
                all_predictions.extend(max_idx.tolist())

        writer.add_scalar("Val/Loss", np.mean(losses), epoch)
        acc = accuracy_score(all_gts, all_predictions)
        writer.add_scalar("Val/Accuracy", acc, epoch)      
        conf_matrix = confusion_matrix(all_gts, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.categories))], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size
        }

        torch.save(checkpoint, os.path.join(args.trained_path, "last.pt"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.trained_path, "best.pt"))
            best_acc = acc

if __name__ == '__main__':
    args = get_args()
    train(args)