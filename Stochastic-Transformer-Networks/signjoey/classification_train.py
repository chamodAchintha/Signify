import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from signjoey.helpers import load_config, set_seed
from signjoey.classification_model import ClassificationModel
from signjoey.builders import build_optimizer
from signjoey.early_stopping import EarlyStopping


def train_model(cfg_file: str, train_loader, val_loader):

    cfg = load_config(cfg_file)
    train_config = cfg['training']

    # output path
    os.makedirs(train_config["model_dir"], exist_ok=True)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    model = ClassificationModel(cfg)

    use_cuda = cfg["training"].get("use_cuda", False)
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    model.to(device)

    # optimization
    current_lr = train_config["learning_rate"]
    optimizer = build_optimizer(
        config=train_config, parameters=model.parameters()
    )

    criterion = torch.nn.CrossEntropyLoss()

    # learning rate scheduling
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        verbose=False,
        threshold_mode="abs",
        factor=train_config.get("decrease_factor", 0.1),
        patience=train_config.get("patience", 5),
        min_lr=train_config.get("learning_rate_min", 1e-6)
    )

    # # Early stopping
    # early_stopping = EarlyStopping(
    #     patience=train_config.get("es_patience", 10),
    #     min_delta=train_config.get("es_min_delta", 0.0)
    # )

    # shuffle = train_config.get("shuffle", True)
    num_epochs = train_config["epochs"]
    best_val_loss = float('inf')
    # batch_size = train_config["batch_size"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward() 

            optimizer.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f} lr: {current_lr}')

        # save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            torch.save(checkpoint, os.path.join(train_config["model_dir"], 'best_model.pth'))
            print(f"New best model saved with validation loss {best_val_loss:.4f}")


        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check for early stopping
        # early_stopping(avg_val_loss)
        # if early_stopping.early_stop:
        #     print(f"Early stopping after {epoch + 1} epochs. Best Validation Loss: {early_stopping.best_loss:.4f}")
        #     break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_loss = criterion(output, target)
            total_val_loss += val_loss.item()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_val_loss, accuracy