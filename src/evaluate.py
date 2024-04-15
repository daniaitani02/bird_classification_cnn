import torch
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.model import build_model  # Import the build_model function to reconstruct the model architecture

def evaluate(model_path: str, test_dir: str, img_height: int = 224, img_width: int = 224, batch_size: int = 32) -> None:
    """
    Evaluates the trained model on the test dataset using PyTorch.

    Args:
        model_path (str): Path to the saved model.
        test_dir (str): Path to the test directory.
        img_height (int): Height of the input images. Default is 224.
        img_width (int): Width of the input images. Default is 224.
        batch_size (int): Size of the batches of data. Default is 32.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations for the test data
    test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Load the test data
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Rebuild the model architecture and load the weights
    model = build_model(num_classes=525)  # Ensure this matches the number of classes used during training
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Initialize loss and accuracy tracking
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0

    # Evaluate the model
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}")

if __name__ == '__main__':
    evaluate(model_path='models/bird_classification_model.pth', test_dir=os.path.join('data', 'test'))
