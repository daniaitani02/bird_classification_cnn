import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
from src.model import build_model, get_optimizer, get_loss_function
from torch.utils.tensorboard import SummaryWriter
import os

def train(model: nn.Module, 
          train_loader: DataLoader,
          test_loader: DataLoader,
          num_classes: int = 525,
          device: str = "cuda",
          epochs: int = 2, 
          experiment_name: str = "Bird_Classification") -> None:
    """
    Trains the bird classification model, tracks the training process using MLflow and TensorBoard,
    and saves the model for later inference.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        num_classes (int): Number of classes in the dataset. Default is 525.
        epochs (int): Number of epochs to train the model. Default is 10.
    """
    # Set up MLflow tracking
    mlflow.set_experiment(experiment_name)

    # Build the model
    model = model.to(device)

    # Define training parameters
    optimizer = get_optimizer(model)
    criterion = get_loss_function()

    # Set up TensorBoard writer
    writer = SummaryWriter()
    # print("Number of training batches:", len(train_loader))
    # print("Number of testing batches:", len(test_loader))
    
    if len(train_loader) == 0 or len(test_loader) == 0:
        raise ValueError("DataLoaders are empty. Check your dataset and paths.")

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.log_params({
            "num_classes": num_classes,
            "epochs": epochs,
            "device": device
        })
        
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch+1}")
            model.train()
            running_loss = 0.0
            # try:
            for batch_idx, (images, labels) in enumerate(train_loader):
                # print(f"Processing batch {batch_idx+1}/{len(train_loader)}")
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Log training loss for each batch
                writer.add_scalar('Batch/Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
                
            # except Exception as e:
                # print("An error occurred during training:", e)
                # raise
                
            # Log average training loss for the epoch
            avg_train_loss = running_loss / len(train_loader)
            writer.add_scalar('Epoch/Average Training Loss', avg_train_loss, epoch)
            
            # Evaluate the model on the test set
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Log test loss and accuracy for the epoch
            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = correct / total
            writer.add_scalar('Epoch/Test Loss', avg_test_loss, epoch)
            writer.add_scalar('Epoch/Test Accuracy', test_accuracy, epoch)
            mlflow.log_metric("Test Accuracy", test_accuracy, step=epoch)
            # print('ending soon....')

            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Save the trained model for later inference        
        model_save_path = os.path.join('models', 'bird_classification_model.pth')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

        # Log the model as an MLflow artifact
        mlflow.log_artifact(model_save_path)

        # Close the TensorBoard writer
        writer.close()
