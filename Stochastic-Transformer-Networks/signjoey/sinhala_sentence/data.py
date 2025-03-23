import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast
import gzip, pickle

TOKENIZER = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", tgt_lang="si_LK")

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def create_target_token_ids(input_ids: torch.Tensor, pad_token_id: int):
    """
    Create target token IDs for decoder.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    language_token_id = prev_output_tokens[:, 0].clone()
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, :index_of_eos] = prev_output_tokens[:, 1:index_of_eos+1].clone()
    prev_output_tokens[:, index_of_eos] = language_token_id

    return prev_output_tokens

class SinhalaSignDataset(Dataset):
    def __init__(self, data_path, tokenizer, decoder_max_len=50):
        """
        Args:
            data_list (list of dicts): Dataset entries, each containing keypoints and target sentence.
            tokenizer: MBart tokenizer for Sinhala sentence tokenization.
            max_length (int): Maximum sequence length for tokenized sentences.
        """
        self.data = load_dataset_file(data_path)
        self.tokenizer = tokenizer
        self.encoder_max_length = max([entry["keypoints"].size(0) for entry in self.data])
        self.decoder_max_length = decoder_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Extract keypoints (encoder inputs)
        keypoints = entry["keypoints"].clone().detach()  # Shape: (num_frames, keypoint_dim)

        frame_count = keypoints.size(0)
        if frame_count < self.encoder_max_length:
            # Pad with zeros
            padding = torch.zeros(self.encoder_max_length - frame_count, keypoints.size(1))
            keypoints = torch.cat([keypoints, padding], dim=0)
            keypoints_mask = torch.cat([torch.ones(frame_count, dtype=torch.int32), torch.zeros(self.encoder_max_length - frame_count, dtype=torch.bool)])
        elif frame_count == self.encoder_max_length:
            # If no padding is needed
            keypoints_mask = torch.ones(self.encoder_max_length, dtype=torch.int32)
        else:
            raise ValueError(f'frame count ({frame_count}) > sequence length ({self.encoder_max_length})')


        # Tokenize Sinhala sentence
        labels = self.tokenizer(
            text_target=entry["sinhala_sentence"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.decoder_max_length
        )

        return {
            "keypoints": keypoints.float(),  # Convert to float tensor
            "keypoints_mask": keypoints_mask,  # Encoder attention mask
            "text_input_ids": labels.input_ids,  # Target tokens
            "text_attention_mask": labels.attention_mask[0],
            "label": create_target_token_ids(labels.input_ids, self.tokenizer.pad_token_id)
        }

def load_training_data(cfg, logger):
    train_data_path = cfg['data']['train_data_path']
    dev_data_path = cfg['data']['dev_data_path']
    decoder_max_len = cfg['data']['max_sent_length']
    batch_size = cfg['data']['batch_size']

    train_dataset = SinhalaSignDataset(train_data_path, TOKENIZER, decoder_max_len)
    dev_dataset = SinhalaSignDataset(dev_data_path, TOKENIZER, decoder_max_len)

    logger.info(f"train dataset size : {len(train_dataset)}")
    logger.info(f"dev dataset size : {len(dev_dataset)}")
    logger.info(f"batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=False)

    return train_loader, dev_loader


def load_test_data(cfg, logger):
    test_data_path = cfg['data']['test_data_path']
    decoder_max_len = cfg['data']['max_sent_length']
    batch_size = cfg['data']['batch_size']

    test_dataset = SinhalaSignDataset(test_data_path, TOKENIZER, decoder_max_len)

    logger.info(f"test dataset size : {len(test_dataset)}")
    logger.info(f"batch size: {batch_size}")

    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return test_loader