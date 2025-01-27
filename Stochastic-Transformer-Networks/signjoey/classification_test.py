import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
import os
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from signjoey.helpers import load_config, make_logger, set_seed
from signjoey.classification_model import ClassificationModel
from signjoey.builders import build_optimizer
from signjoey.early_stopping import EarlyStopping
from signjoey.classification_data import load_test_data
from sklearn.preprocessing import LabelEncoder


def test_model(cfg_file: str):
    cfg = load_config(cfg_file)

    logger = make_logger(model_dir=cfg['training']["model_dir"], log_file=f"{cfg['name']}_test.log")

    set_seed(seed=cfg["training"].get("random_seed", 42))

    # Load the checkpoint
    checkpoint_path = os.path.join(cfg["training"]["model_dir"], 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    logger.info(f"checkpoint loaded from {checkpoint_path}")

    # Load test data
    label_encoder = checkpoint['label_encoder']
    test_loader = load_test_data(cfg, label_encoder=label_encoder)

    # Load model
    model = ClassificationModel(cfg, logger)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Classification model loaded.")
    
    use_cuda = cfg["training"].get("use_cuda", False)
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    logger.info(f"device: {device}")
    model.to(device)

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_test_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target, mask) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            mask = mask.unsqueeze(1).expand(-1, 1, -1)
            data, target, mask = data.to(device), target.to(device), mask.to(device)

            output = model(data, mask)
            test_loss = criterion(output, target)
            total_test_loss += test_loss.item()

            # Predictions and actual targets
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    # Calculate metrics
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')

    # Print and save results
    logger.info(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    # Save predictions to CSV
    df = pd.DataFrame({
        'True_Class': all_targets,
        'Pred_Class': all_preds,
        'True_Label': label_encoder.inverse_transform(all_targets),
        'Pred_Label': label_encoder.inverse_transform(all_preds)
    })
    df.to_csv(f"{cfg['training']['model_dir']}/{cfg['name']}_test_results.csv", index=False)


