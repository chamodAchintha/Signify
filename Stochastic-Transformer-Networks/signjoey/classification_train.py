import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from signjoey.helpers import load_config, make_logger, set_seed
from signjoey.classification_model import ClassificationModel
from signjoey.builders import build_optimizer
from signjoey.early_stopping import EarlyStopping
from signjoey.classification_data import load_training_data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pandas as pd

def train_model(cfg_file: str):

    cfg = load_config(cfg_file)
    train_config = cfg['training']

    # output path
    os.makedirs(train_config["model_dir"], exist_ok=True)

    logger = make_logger(model_dir=train_config["model_dir"], log_file=f"{cfg['name']}_train.log")
    validation_file = f"{train_config['model_dir']}/{cfg['name']}_validation.txt"
    with open(validation_file, "w", encoding="utf-8") as opened_file:
        pass

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    label_encoder = LabelEncoder()
    train_loader, val_loader = load_training_data(cfg, label_encoder)    
    logger.info(f"train loader size: {len(train_loader)}\nvalidtion loader size: {len(val_loader)}")

    model = ClassificationModel(cfg, logger)
    logger.info(f'classification model created:\n{model}')

    use_cuda = cfg["training"].get("use_cuda", False)
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    logger.info(f"device: {device}")

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

        for batch_idx, (data, target, mask) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            mask = mask.unsqueeze(1).expand(-1, 1, -1)
            data, target, mask = data.to(device), target.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, target)
            loss.backward() 

            optimizer.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_val_loss, val_accuracy, val_f1, all_preds, all_targets = validate_model(model, val_loader, criterion, device)

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f} F1: {val_f1:.4f} lr: {current_lr}')
        with open(validation_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f} F1: {val_f1:.4f} lr: {current_lr}\n')

        # save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'Accuracy': val_accuracy,
                'F1': val_f1,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'label_encoder': label_encoder
            }
            torch.save(checkpoint, os.path.join(train_config["model_dir"], 'best_model.pth'))
            logger.info(f"New best model saved with validation loss {best_val_loss:.4f}")

            df = pd.DataFrame({
            'True_Label': all_targets,
            'Pred_Label': all_preds
            })
            # Save to CSV
            df.to_csv(f"{train_config['model_dir']}/{cfg['name']}_validation_results.csv", index=False)


        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check for early stopping
        # early_stopping(avg_val_loss)
        # if early_stopping.early_stop:
        #     print(f"Early stopping after {epoch + 1} epochs. Best Validation Loss: {early_stopping.best_loss:.4f}")
        #     break

        # if device.type == 'cuda':
        #     torch.cuda.empty_cache()
    logger.info('Training Completed.')

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target, mask) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation"):
            mask = mask.unsqueeze(1).expand(-1, 1, -1)
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            # print('data shape:', data.shape)
            output = model(data, mask)
            # print("Validatation output shape:", output.shape)
            val_loss = criterion(output, target)
            total_val_loss += val_loss.item()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect predictions and targets for F1 score
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    # print(all_preds)
    # print(all_targets)
    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return avg_val_loss, accuracy, f1, all_preds, all_targets