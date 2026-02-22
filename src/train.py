import torch
import mlflow
import mlflow.pytorch
from torch import nn, optim
from tqdm import tqdm
import json

from sklearn.metrics import confusion_matrix

from config import Config
from preprocess import get_train_transforms, get_val_transforms
from dataset import create_dataloader
from model import LogisticRegressionModel


def train():

    cfg = Config()
    mlflow.set_tracking_uri("file:./mlruns")

    # ---------------------------
    # DATALOADERS
    # ---------------------------
    train_loader = create_dataloader(
        cfg.train_dir,
        get_train_transforms(cfg.image_size),
        cfg.batch_size,
        shuffle=True
    )

    val_loader = create_dataloader(
        cfg.val_dir,
        get_val_transforms(cfg.image_size),
        cfg.batch_size,
        shuffle=False
    )

    # Save class names (important for inference)
    class_names = train_loader.dataset.classes
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # MODEL
    # ---------------------------
    input_dim = cfg.image_size * cfg.image_size * 3

    model = LogisticRegressionModel(
        input_dim=input_dim,
        num_classes=cfg.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.lr
    )

    mlflow.set_experiment("cats_dogs_classifier")

    # ---------------------------
    # TRAINING
    # ---------------------------
    run_name = f"logreg_img{cfg.image_size}_bs{cfg.batch_size}_lr{cfg.lr}"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("batch_size", cfg.batch_size)
        mlflow.log_param("learning_rate", cfg.lr)
        mlflow.log_param("epochs", cfg.epochs)

        for epoch in range(cfg.epochs):

            model.train()
            running_loss = 0.0

            loop = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{cfg.epochs}"
            )

            for images, labels in loop:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)

            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

            # save metrics
            with open("metrics.json", "w") as f:
                json.dump({"loss": avg_loss}, f)

            mlflow.log_metric("loss", avg_loss, step=epoch)

        # ---------------------------
        # VALIDATION (CONFUSION MATRIX)
        # ---------------------------
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)

        with open("confusion_matrix.json", "w") as f:
            json.dump(cm.tolist(), f)

        torch.save(model.state_dict(), cfg.model_path)
        # ---------------------------
        # LOG ARTIFACTS
        # ---------------------------
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("confusion_matrix.json")
        mlflow.log_artifact("class_names.json")
        mlflow.log_artifact("models/model.pt")

        # save model
        mlflow.pytorch.log_model(model, "LogisticRegression Model")


if __name__ == "__main__":
    train()
