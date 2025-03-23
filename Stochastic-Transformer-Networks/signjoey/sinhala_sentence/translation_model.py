import torch.nn as nn
from torch import Tensor
import os
from signjoey.helpers import load_checkpoint
from signjoey.embeddings import SpatialEmbeddings
from signjoey.encoders import TransformerEncoder
from signjoey.classification_head import MLPHead, ConvHead, RNNHead, AttentionHead
from signjoey.sinhala_sentence.mbart_decoder import MBartDecoder

class SinhalaSignTranslationModel(nn.Module):
    def __init__(self, cfg, logger):
        """
        Initializes the classification model with an encoder and a classification head.
        Parameters:
        - encoder: The encoder model (e.g., the encoder part of a transformer)
        - head_type: The type of classification head (e.g., 'mlp', 'attention', 'conv', 'rnn')
        - config: A dictionary containing configuration parameters for the classification head
        """
        super(SinhalaSignTranslationModel, self).__init__()

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

        # mbart deocder
        self.decoder = MBartDecoder(self, cfg, logger)
        

    def forward(
        self, 
        sgn: Tensor, 
        sgn_mask: Tensor,
        text_input_ids: Tensor,
        text_attention_mask: Tensor,
        label: Tensor
    ):
        """
        Forward pass for Sinhala Sign Language translation.
        """
        # Compute sign embeddings
        sgn_embedded = self.sgn_embed(sgn, sgn_mask)
        # Encode sign language representations
        encoder_output = self.encoder(sgn_embedded, sgn_mask)

        # Pass encoder output to decoder
        decoder_output = self.decoder(
            encoder_attention_mask=sgn_mask,
            encoder_outputs=(encoder_output,),
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask,
            labels=label
        )
        return decoder_output

