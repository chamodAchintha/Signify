import torch.nn as nn
import os
from signjoey.helpers import load_checkpoint
from signjoey.embeddings import SpatialEmbeddings
from signjoey.encoders import TransformerEncoder
from signjoey.classification_head import MLPHead, ConvHead, RNNHead, AttentionHead
from signjoey.sinhala_sentense.mbart_decoder import MBartDecoder

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
        

    def forward(self, x, mask):
        """
        Forward pass through the model.
        Parameters:
        - x: Input tensor to the model
        Returns:
        - Output tensor after passing through encoder and classification head
        """
        x = self.sgn_embed(x, mask)
        x = self.encoder(x, mask)[0]
        x = self.decoder(
            encoder_attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        )
        return x

