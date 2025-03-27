import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput, BaseModelOutput
from signjoey.helpers import freeze_params


from typing import List, Optional, Tuple, Union, Any

# Load updated mBART model and tokenizer
MODEL_NAME = "facebook/mbart-large-50"
MODEL = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class MBartDecoder(nn.Module):
    def __init__(self, cfg, logger: Optional[Any] = None):
        super().__init__()

        self.decoder= MODEL.model.decoder
        self.lm_head = MODEL.lm_head
        self.final_logits_bias = MODEL.final_logits_bias
        
        self.mbart_config = MODEL.model.config
        self.decoder_config = MODEL.model.decoder.config
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1
        self.d_model = 1024
        self.decoder_attention_heads= 16
        self.decoder_layers = 12
        self.decoder_start_token_id = 2        
        self.max_length = 200
        self.max_position_embeddings= 1024
        self.num_beams= 5
        self.tokenizer_class= "MBart50Tokenizer"
        self.vocab_size= 250054

        self.logger = logger
        self.inference_sample_size=cfg['model']['inference_sample_size']

        if cfg['model']['decoder'].get('freeze', True):
            freeze_params(self)
            logger.info('Freezed all decoder layers')

        num_layers_to_train = cfg['model']['decoder'].get('num_layers_to_train', 12)
        if num_layers_to_train > self.mbart_config.decoder_layers:
            logger.warn(f"number of decoder layers to train ({num_layers_to_train}) is greater than the number of layers. Train all layers ({self.mbart_config.decoder_layers})")
            num_layers_to_train = self.mbart_config.decoder_layers

        train_layers = cfg['model']['decoder'].get('train_layers', [])

        for i in range(num_layers_to_train):
            for name, param in self.decoder.layers[i].named_parameters():
                if name.split('.')[0] in train_layers:
                    param.requires_grad = True
                    logger.info(f"Decoder layer - {i} - {name} is set to train")

    # This is if inference_samples from stochastic encoder are present
    # def forward(
    #     self,
    #     encoder_attention_mask: Optional[torch.Tensor] = None,
    #     decoder_input_ids: Optional[torch.LongTensor] = None,
    #     decoder_attention_mask: Optional[torch.LongTensor] = None,
    #     encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    # ):
    #     if self.training:
    #         return self.forward_(
    #             encoder_attention_mask=encoder_attention_mask,
    #             encoder_outputs=encoder_outputs,
    #             decoder_input_ids=decoder_input_ids,
    #             decoder_attention_mask=decoder_attention_mask,
    #             labels=labels
    #         )
    #     else:
    #         logits = None
    #         output = None
    #         encoder_s = encoder_outputs[0].shape[-1]
    #         inference_sample_size= max(self.inference_sample_size, encoder_s)

    #         for i in range(inference_sample_size):
    #             logits_, output_ = self.forward_(
    #                 encoder_attention_mask=encoder_attention_mask,
    #                 encoder_outputs=(encoder_outputs[0][...,i%encoder_s],),
    #                 decoder_input_ids=decoder_input_ids,
    #                 decoder_attention_mask=decoder_attention_mask,
    #                 labels=labels
    #             )
    #             if logits is None:
    #                 logits = logits_
    #                 output = output_
    #             else:
    #                 logits += logits_
    #                 output += output_

                
    #         output=output*1.0/inference_sample_size
    #         logits=logits*1.0/inference_sample_size

    #         return logits, output

    # if inference_samples from stochastic encoder are present, make this as forward_ function
    def forward(
        self,
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
    ):
        if encoder_outputs is None:
            raise ValueError("Encoder outputs cannot be None. Ensure the encoder is correctly producing outputs before passing them to the decoder.")

        
        return_dict = self.mbart_config.use_return_dict

        if labels is not None:
            if use_cache:
                if self.logger:
                    self.logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
                else: 
                    print("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # if decoder_input_ids is None and decoder_inputs_embeds is None:
            #     decoder_input_ids = shift_tokens_right(labels, MODEL.config.pad_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.mbart_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.mbart_config.output_hidden_states
        )

        # encoder_output.last_hidden_state.shape = torch.Size([1, 7, 1024])
        if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
        else:
            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias.to(outputs[0].device) # add to device separately since final_logits_bias seems not adding to device

        # masked_lm_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.mbart_config.vocab_size), labels.view(-1))

        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return lm_logits, outputs.last_hidden_state

        # return Seq2SeqLMOutput(
        #     loss=masked_lm_loss,
        #     logits=lm_logits,
        #     past_key_values=outputs.past_key_values,
        #     decoder_hidden_states=outputs.decoder_hidden_states,
        #     decoder_attentions=outputs.decoder_attentions,
        #     cross_attentions=outputs.cross_attentions,
        #     encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        #     encoder_hidden_states=outputs.encoder_hidden_states,
        #     encoder_attentions=outputs.encoder_attentions,
        # )