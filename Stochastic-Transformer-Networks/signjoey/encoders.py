# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from signjoey.helpers import freeze_params
from signjoey.transformer_layers import TransformerEncoderLayer, PositionalEncoding
from signjoey.layers import DenseBayesian

# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs
    ) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """

        super(RecurrentEncoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=invalid-name, unused-argument
    def _check_shapes_input_forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param embed_src: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(
            embed_src=embed_src, src_length=src_length, mask=mask
        )

        # apply dropout to the rnn input
        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length, batch_first=True)
        output, hidden = self.rnn(packed)

        # pylint: disable=unused-variable
        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)


    
class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.2,
        emb_dropout: float = 0.2,
        freeze: bool = False,
        skip_encoder: bool = False,
        bayesian_attention: bool = False,
        bayesian_feedforward: bool = False,
        ibp: bool = False,
        inference_sample_size: int = 10,
        activation: str ='relu',
        lwta_competitors: int = 4,
        edge_index= torch.tensor([[0, 6], [0, 5], [6, 8], [5, 7], [0, 62], [62, 63], [63, 64], [59, 64], [59, 60], [60, 61], [61, 62], [0, 74], [71, 72], [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [71, 82], [71, 83], [77, 87], [83, 84], [84, 85], [85, 86], [86, 87], [87, 88], [88, 89], [89, 90], [83, 90], [0, 65], [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [65, 70], [7, 91], [91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100], [100, 101], [101, 102], [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110], [110, 111], [8, 112], [112, 113], [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121], [121, 122], [122, 123], [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131], [131, 132]]).t().contiguous(),

        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param bayesian_attention: using or not gaussian weights 
        :param bayesian_feedforward: using or not gaussian weights 
        :param ibp: using or not ibp method
        :param inference_sample_size: sample for bayesian averaging
        :param lwta_competitors: the size of LWTA's competion block (U parameter)
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()
        if skip_encoder:
            bayesian_attention=False
            bayesian_feedforward=False
        
        self.bayesian = bayesian_attention or bayesian_feedforward 
        if self.bayesian:
            self.inference_sample_size=inference_sample_size
        else :
            self.inference_sample_size=1
        
        self.lwta_competitors=lwta_competitors
        
        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    edge_index=edge_index,
                    
                    bayesian_attention=bayesian_attention,
                    bayesian_feedforward=bayesian_feedforward,
                    ibp=ibp,
                    activation=activation,
                    lwta_competitors=lwta_competitors
                )
                for _ in range(num_layers)
            ]
        )
        
        self.skip_encoder=skip_encoder
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
       
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        if freeze:
            freeze_params(self)
            
   
    # # pylint: disable=arguments-differ
    # def forward(
    #     self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    # ) -> (Tensor, Tensor):
    #     if self.training:
    #         return self.forward_(embed_src,src_length,mask)
    #     else:
            
    #         out=[]
    #         embed_s=embed_src.shape[-1]
    #         inference_sample_size= max(self.inference_sample_size,embed_s)
    #         for i in range(inference_sample_size):
               
    #            x_, _=  self.forward_(embed_src[...,i%embed_s],src_length,mask)
               
    #            out.append(torch.unsqueeze(x_,-1))
    #     out=torch.cat(out,-1)
      
 
        
    #     return out, None

    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor, edge_index: Tensor
    ) -> (Tensor, Tensor):
        """
        Forward pass with GCN integration.
        :param embed_src: Embedded source inputs (batch_size, src_len, embed_size)
        :param src_length: Length of source inputs (batch_size,)
        :param mask: Attention mask (batch_size, src_len, src_len)
        :param edge_index: Graph edges for GCN (required by each layer)
        :return: Encoded output and None
        """
        if self.training:
            return self.forward_(embed_src, src_length, mask, edge_index)
        else:
            out = []
            embed_s = embed_src.shape[-1]
            inference_sample_size = max(self.inference_sample_size, embed_s)

            for i in range(inference_sample_size):
                x_, _ = self.forward_(
                    embed_src[..., i % embed_s], src_length, mask, edge_index
                )
                out.append(torch.unsqueeze(x_, -1))

            out = torch.cat(out, -1)
            return out, None

    
    # #hidden forward (single run)
    # def forward_(
    #     self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    # ) -> (Tensor, Tensor):
    #     """
    #     Pass the input (and mask) through each layer in turn.
    #     Applies a Transformer encoder to sequence of embeddings x.
    #     The input mini-batch x needs to be sorted by src length.
    #     x and mask should have the same dimensions [batch, time, dim].

    #     :param embed_src: embedded src inputs,
    #         shape (batch_size, src_len, embed_size)
    #     :param src_length: length of src inputs
    #         (counting tokens before padding), shape (batch_size)
    #     :param mask: indicates padding areas (zeros where padding), shape
    #         (batch_size, src_len, embed_size)
    #     :return:
    #         - output: hidden states with
    #             shape (batch_size, max_length, directions*hidden),
    #         - hidden_concat: last hidden state with
    #             shape (batch_size, directions*hidden)
    #     """

    #     x = embed_src
    #     x = self.pe(x)  # add position encoding to word embeddings
    #     x = self.emb_dropout(x)
    #     if not self.skip_encoder:
    #         for layer in self.layers:
    #             x = layer(x,mask )
        
       
    #     x=self.layer_norm(x)
    #     return x, None

    # def __repr__(self):
    #     return "%s(num_layers=%r, num_heads=%r)" % (
    #         self.__class__.__name__,
    #         len(self.layers),
    #         self.layers[0].src_src_att.num_heads,
    #     )
    

    def forward_(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor, edge_index: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :param edge_index: graph edges for GCN processing
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = embed_src
        x = self.pe(x)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        if not self.skip_encoder:
            for layer in self.layers:
                # Pass edge_index to each TransformerEncoderLayer
                x = layer(x, mask, edge_index)

        x = self.layer_norm(x)  # Normalize final output
        return x, None

