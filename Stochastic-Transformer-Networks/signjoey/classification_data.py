import pickle
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

class SignLanguageDataset(Dataset):
    def __init__(self, data_path, seq_length, label_encoder=None, train=False, transform=None):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.sequence_length = seq_length
        self.transform = transform
        self.label_encoder = label_encoder

        # Initialize and fit the LabelEncoder on all unique classes in the dataset
        if self.label_encoder is None:
            raise ValueError('No label encoder provided')
        if train:
            self.label_encoder.fit([entry['sinhala_word'] for entry in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the data entry at the specified index
        entry = self.data[idx]
        
        keypoints = entry['keypoints'].clone().detach()

        frame_count = keypoints.size(0)
        if frame_count < self.sequence_length:
            # Pad with zeros
            padding = torch.zeros(self.sequence_length - frame_count, keypoints.size(1))
            keypoints = torch.cat([keypoints, padding], dim=0)
            mask = torch.cat([torch.ones(frame_count, dtype=torch.bool), torch.zeros(self.sequence_length - frame_count, dtype=torch.bool)])
        elif frame_count == self.sequence_length:
            # If no padding is needed
            mask = torch.ones(self.sequence_length, dtype=torch.bool)
        else:
            raise ValueError(f'frame count ({frame_count}) > sequence length ({self.sequence_length})')


        target_class = self.label_encoder.transform([entry['sinhala_word']])[0]
        target_class = torch.tensor(target_class, dtype=torch.long) 

        # Apply any transforms if provided
        if self.transform:
            keypoints = self.transform(keypoints)

        target_embed = entry['embedding']

        return keypoints, target_class, mask, target_embed



def load_training_data(cfg, label_encoder):
    
    dataset = SignLanguageDataset(cfg['data']['train_data_path'], cfg['data']['seq_length'],train=True, label_encoder=label_encoder)
    all_indices = list(range(len(dataset)))
    all_labels = [dataset[i][1] for i in all_indices]  # assuming labels are the second element in each dataset item

    # Stratified split for train and validation
    train_indices, val_indices = train_test_split(
        all_indices, test_size=cfg['data']['val_split'], stratify=all_labels, random_state=cfg['training']['random_seed']
    )

    # Subset the dataset using the indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    batch_size = cfg['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader


def load_test_data(cfg, label_encoder):
    test_dataset = SignLanguageDataset(cfg['data']['test_data_path'], cfg['data']['seq_length'], train=False, label_encoder=label_encoder)

    # Create data loaders
    batch_size = cfg['data']['batch_size']
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return test_loader