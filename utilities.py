# Import necessary libraries
import argparse

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend so plots can be saved in a codespace
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# Use the argparse library to take inputs from the user
def get_args():
    """Parse the command line arguments: num_epochs, batch_size, learning_rate."""
    parser = argparse.ArgumentParser(
        description='Train a CNN on the MNIST dataset with PyTorch.')
    parser.add_argument('-e', '--num_epochs', type=int, default=2,
                        help='Number of epochs (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='Batch size (default: 1000)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    return parser.parse_args()


# Convolutional neural network definition
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)  # First convolutional layer
        out = self.layer2(out)  # Second convolutional layer
        out = out.reshape(out.size(0), -1)  # Flatten for the fully connected layers
        out = self.fc1(out)  # First fully connected layer
        out = self.fc2(out)  # Second fully connected layer
        return out


class MNISTTrainer:
    # Hyperparameters
    # num_epochs    =  Number of times the entire dataset is passed through the model
    # batch_size    =  Number of samples per batch to be passed through the model
    # learning_rate =  Step size for parameter updates
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Device configuration (use GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    # Track accuracy of the model
    def track_accuracy(self, model, loader):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(self.device)  # Move images to the configured device
                labels = labels.to(self.device)  # Move labels to the configured device
                outputs = model(images)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total += labels.size(0)  # Increment total by the number of labels
                correct += (predicted == labels).sum().item()  # Count correct predictions
            accuracy = 100 * correct / total  # Calculate accuracy
        model.train()  # Set the model back to training mode
        return accuracy

    # Training the model
    def train(self, model, train_loader, test_loader, criterion, optimizer):
        loss_list = []  # Average loss of each epoch
        acc_train = []  # Training accuracy of each epoch
        acc_test = []  # Testing accuracy of each epoch

        # Loop over the number of epochs
        for epoch in range(self.num_epochs):
            # Loop over each batch in the training DataLoader
            pbar = tqdm(train_loader,
                        desc=f"Training: {epoch + 1:03d}/{self.num_epochs}",
                        ncols=125,
                        leave=True)

            running_loss = []  # Loss of each batch in this epoch

            for i, (images, labels) in enumerate(pbar, 1):
                # Images are (batch_size, channels, height, width), e.g., (1000, 1, 28, 28)
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass and loss
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss.append(loss.item())

                # Backward pass & optimize
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                # Show the running average loss
                pbar.set_postfix(loss=f"{sum(running_loss) / i : 10.6f}")

            # Average of all batch losses is the loss of the current epoch
            loss_list.append(sum(running_loss) / len(running_loss))

            # Track accuracy on the train and test sets
            acc_train.append(self.track_accuracy(model, train_loader))
            acc_test.append(self.track_accuracy(model, test_loader))

        return model, loss_list, acc_train, acc_test

    # Plot results
    # I extended the plotting function because the assignment provided one
    #   did not include some axis labels and the second plot
    @staticmethod
    def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
        # Plot the loss & accuracy curves
        plt.figure(figsize=(10, 4))

        # Plot the training loss over iterations
        plt.subplot(1, 2, 1)
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title("Training Loss")

        # Plot the train & test accuracies over epochs
        plt.subplot(1, 2, 2)
        plt.plot(acc_train)
        plt.plot(acc_test)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Model Accuracy")
        plt.legend(['Training', 'Testing'])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Figure saved to {save_path}")

    # Set up the data, model, and trainer for the main run
    def main_run(self):
        # MNIST dataset
        # Download & load the training dataset, converting images to PyTorch tensors
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        # Download & load the test dataset, converting images to PyTorch tensors
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  transform=transforms.ToTensor(),
                                                  download=True)

        # Data loaders: batching and shuffling
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        # Create the model & move it to the configured device (GPU/CPU)
        model = ConvNet().to(self.device)
        print(model)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Train the model
        model, loss_list, acc_train, acc_test = self.train(
            model, train_loader, test_loader, criterion, optimizer)

        # Make some plots!
        self.plot(loss_list, acc_train, acc_test)
