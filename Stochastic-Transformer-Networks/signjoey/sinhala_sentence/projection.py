import torch
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_d_model, inference_sample_size):
        super().__init__()
    
        self.projection = nn.Linear(encoder_hidden_size,  decoder_d_model)
        self.inference_sample_size = inference_sample_size

    def forward(self, encoder_output):
      if self.training:
        return self.projection(encoder_output)

      else:
        projections = None
        encoder_s = encoder_output.shape[-1]
        inference_sample_size= max(self.inference_sample_size, encoder_s)

        for i in range(inference_sample_size):
            out = self.projection(encoder_output[...,i%encoder_s])

            if projections is None:
                projections = out
            else:
                projections += out

            
        projections=projections*1.0/inference_sample_size
        return projections
