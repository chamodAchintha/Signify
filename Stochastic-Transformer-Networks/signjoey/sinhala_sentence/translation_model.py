import torch.nn as nn
from torch import Tensor
import os
from signjoey.helpers import load_checkpoint
from signjoey.embeddings import SpatialEmbeddings
from signjoey.encoders import TransformerEncoder
from signjoey.classification_head import MLPHead, ConvHead, RNNHead, AttentionHead
from signjoey.sinhala_sentence.mbart_decoder import MBartDecoder
from signjoey.sinhala_sentence.projection import Projection
import torch
from signjoey.helpers import freeze_params

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
        self.logger.info('creating the Translation model...')

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
        
        # load only the encoder and encoder spatial embedding state from checkpoint
        use_encoder_checkpoint = cfg['model']['encoder'].get('use_checkpoint', False)
        if use_encoder_checkpoint:
            use_cuda = cfg["training"].get("use_cuda", False)
            device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
            checkpoint_path = cfg['model']['encoder']['checkpoint']
            if not os.path.exists(checkpoint_path):
                 raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")
            model_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            encoder_state_dict = {k[8:]: v for k, v in model_checkpoint["model_state_dict"].items() if k.startswith('encoder.')}
            embed_state_dict = {k[10:]: v for k, v in model_checkpoint["model_state_dict"].items() if k.startswith('sgn_embed.')}
            self.encoder.load_state_dict(encoder_state_dict)
            self.sgn_embed.load_state_dict(embed_state_dict)
            self.logger.info(f'loaded the embed and encoder state from the checkpoint - {checkpoint_path}')

        # mbart deocder
        self.decoder = MBartDecoder(cfg, logger) 
        self.decoder_config = self.decoder.mbart_config
        
        # map encoder and decoder
        self.projection = Projection(cfg['model']['encoder'].get('hidden_size'), self.decoder_config.d_model, cfg['model']['inference_sample_size'])

        # load full model from a checkpoint
        use_model_checkpoint = cfg['model'].get('use_checkpoint', False)
        self.logger.info(f"Load model weights from a checkpoint: {use_model_checkpoint}")
        if use_model_checkpoint:
            if use_encoder_checkpoint:
                self.logger.warn(f"Already loaded an encoder checkpoint. It will be discarded.")

            use_cuda = cfg["training"].get("use_cuda", False)
            device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")

            checkpoint_path = cfg['model']['checkpoint']
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")
            
            self.logger.info(f"Loading model weights from {checkpoint_path}")
            model_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.load_state_dict(model_checkpoint['model_state_dict'])
            self.logger.info(f"Checkpoint Loaded.")

        self.freeze(cfg)


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
        sgn_embedded = self.sgn_embed(
            sgn, 
            sgn_mask.unsqueeze(1).expand(-1, 1, -1).bool()
        )
        # Encode sign language representations
        encoder_output = self.encoder(
            sgn_embedded, 
            sgn_mask.unsqueeze(1).expand(-1, 1, -1).bool()
        )

        encoder_projection = (self.projection(encoder_output[0]),)

        # Pass encoder output to decoder
        logits, decoder_last_hidden_state = self.decoder(
            encoder_attention_mask=sgn_mask,
            encoder_outputs=encoder_projection,
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask,
            labels=label
        )
        return logits, decoder_last_hidden_state


    def encode(self, sgn: Tensor, sgn_mask: Tensor,):
        # Compute sign embeddings
        sgn_embedded = self.sgn_embed(
            sgn, 
            sgn_mask.unsqueeze(1).expand(-1, 1, -1).bool()
        )
        # Encode sign language representations
        encoder_output = self.encoder(
            sgn_embedded, 
            sgn_mask.unsqueeze(1).expand(-1, 1, -1).bool()
        )

        return (self.projection(encoder_output[0]),)
    
    def decode(self, encoder_output , sgn_mask: Tensor, text_input_ids: Tensor, text_attention_mask: Tensor, label: Tensor):
        return self.decoder(
            encoder_attention_mask=sgn_mask,
            encoder_outputs=encoder_output,
            decoder_input_ids=text_input_ids,
            decoder_attention_mask=text_attention_mask,
            labels=label
        )
    
    def freeze(self, cfg):
        # freeze sign embeddings
        freeze_sign_embed = cfg['model']['encoder']['embeddings'].get('freeze', False)
        if freeze_sign_embed:
            freeze_params(self.sgn_embed)
            self.logger.info('Freezed sign embeddings.')

        # freeze encoder layers
        freeze_encoder = cfg['model']['encoder'].get('freeze', False)
        if freeze_encoder:
            total_encoder_layers = cfg['model']['encoder']['num_layers']
            num_layers_to_freeze = cfg['model']['encoder'].get('num_layers_to_freeze', 0)
            freeze_lower_layers = cfg['model']['encoder'].get('freeze_lower_layers', True)


            if num_layers_to_freeze > total_encoder_layers:
                self.logger.info(f"Encoder has {total_encoder_layers} layers. Asked to freeze {num_layers_to_freeze} layers!")
                self.logger.info('All encoder attention layers will be freezed.')
                num_layers_to_freeze = total_encoder_layers

            if freeze_lower_layers:
                layer_indices = range(num_layers_to_freeze)
            else:
                layer_indices = range(total_encoder_layers - num_layers_to_freeze, total_encoder_layers)


            freeze_layers = cfg['model']['encoder'].get('freeze_layers', [])

            for i in layer_indices:
                for name, param in self.encoder.layers[i].named_parameters():
                    if name.split('.')[0] in freeze_layers:
                        param.requires_grad = False
                        self.logger.info(f"Encoder layer - {i} - {name} is freezed.")
        
        # freeze projection layer
        freeze_projection_layer = cfg['model'].get('freeze_projection_layer', False)
        if freeze_projection_layer:
            freeze_params(self.projection)
            self.logger.info('Freezed projection layer.')

        # freeze decoder layer
        self.decoder.freeze_layers(cfg)
        


