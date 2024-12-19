import torch
import torch.nn as nn
import os
from signjoey.helpers import load_checkpoint
from signjoey.embeddings import SpatialEmbeddings
from signjoey.encoders import TransformerEncoder
from signjoey.classification_head import MLPHead, ConvHead, RNNHead, AttentionHead

class ClassificationModel(nn.Module):
    def __init__(self, cfg, logger):
        """
        Initializes the classification model with an encoder and a classification head.
        Parameters:
        - encoder: The encoder model (e.g., the encoder part of a transformer)
        - head_type: The type of classification head (e.g., 'mlp', 'attention', 'conv', 'rnn')
        - config: A dictionary containing configuration parameters for the classification head
        """
        super(ClassificationModel, self).__init__()

        self.logger = logger
        self.logger.info('creating the classification model...')

        # embeddings
        self.sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
            **cfg['model']["encoder"]["embeddings"],
            num_heads=cfg['model']["encoder"]["num_heads"],
            input_size=cfg["data"]["feature_size"],
            inference_sample_size=cfg['model']['inference_sample_size']
        )

        enc_dropout = cfg['model']["encoder"].get("dropout", 0.0)
        enc_emb_dropout = cfg['model']["encoder"]["embeddings"].get("dropout", enc_dropout)

        # encoder
        self.encoder = TransformerEncoder(
            **cfg['model']["encoder"],
            emb_size=self.sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
            inference_sample_size=cfg['model']['inference_sample_size']
        )
        
        # load encoder and spatial embedding state from checkpoint
        use_checkpoint = cfg['model']['encoder'].get('use_checkpoint', False)
        if use_checkpoint:
            use_cuda = cfg["training"].get("use_cuda", False)
            checkpoint_path = cfg['model']['encoder']['checkpoint']
            if not os.path.exists(checkpoint_path):
                 raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")
            model_checkpoint = load_checkpoint(checkpoint_path, use_cuda=use_cuda)
            encoder_state_dict = {k[8:]: v for k, v in model_checkpoint["model_state"].items() if k.startswith('encoder.')}
            embed_state_dict = {k[10:]: v for k, v in model_checkpoint["model_state"].items() if k.startswith('sgn_embed.')}
            self.encoder.load_state_dict(encoder_state_dict)
            self.sgn_embed.load_state_dict(embed_state_dict)
            self.logger.info(f'loaded the embed and encoder state from the checkpoint - {checkpoint_path}')

        # classification head
        head_type = cfg['model']['classification_head']['type']
        if head_type == 'mlp':
            self.head = MLPHead(cfg)
        elif head_type == 'attention':
            self.head = AttentionHead(cfg)
        elif head_type == 'conv':
            self.head = ConvHead(cfg)
        elif head_type == 'rnn':
            self.head = RNNHead(cfg)
        else:
            raise ValueError(f"Unsupported classification head type: {head_type}")

    def forward(self, x, mask):
        """
        Forward pass through the model.
        Parameters:
        - x: Input tensor to the model
        Returns:
        - Output tensor after passing through encoder and classification head
        """
        x = self.sgn_embed(x, mask)
        x_encoder = self.encoder(x, mask)[0]
        x, emb = self.head(x_encoder)

        return x, emb

        # if self.training:
        #     return x, x_encoder
        # else:
        #     return x, torch.stack(x_encoder).mean(dim=0)

