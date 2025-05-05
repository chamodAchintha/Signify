import math
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from signjoey.helpers import freeze_params
from signjoey.layers import  DenseBayesian,EmbeddingBayesian
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data

def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))


class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])




class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 8,
        scale: bool = False,
        scale_factor: float = None,
        norm_type: str = None,
        activation_type: str = 'relu',
        lwta_competitors: int = 4,
        vocab_size: int = 0,
        padding_idx: int = 1,
        freeze: bool = False,
        bayesian : bool = False,
        inference_sample_size : int = 1,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()
        
        self.bayesian=bayesian    
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        if bayesian:
            self.inference_sample_size=inference_sample_size
            self.lut = EmbeddingBayesian(vocab_size, self.embedding_dim, padding_idx=padding_idx,
                            input_features=vocab_size, output_features=self.embedding_dim, competitors=4,
                 activation='lwta',kl_w=0.1)
        else:
            self.inference_sample_size=1
            self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)
                        

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type and not self.bayesian :
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward_(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param mask: token masks
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """

        x = self.lut(x)
       
        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type and not self.bayesian:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.vocab_size,
        )
    # pylint: disable=arguments-differ
    def forward( self, x: Tensor, mask: Tensor = None) -> Tensor:
        if self.training :
            return self.forward_(x,mask)
        else:
            
            out=[]
            #for i in range(self.inference_sample_size):
            for i in range(self.inference_sample_size):
               
               x_=  self.forward_(x,mask)
               
               out.append(torch.unsqueeze(x_,-1))
        out=torch.cat(out,-1)
      
        
        
        return out

    
class SpatialEmbeddings(nn.Module):

   
    # pylint: disable=unused-argument
    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        lwta_competitors: int = 4,
        scale: bool = False,
        scale_factor: float = None,
        bayesian : bool = False,
        ibp : bool = False,
        inference_sample_size : int = 1,
        **kwargs
    ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()
        

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.bayesian=bayesian
        if self.bayesian:
            self.inference_sample_size=inference_sample_size
        else:
            self.inference_sample_size=1
        if bayesian:
            self.ln = DenseBayesian(self.input_size, self.embedding_dim, competitors =lwta_competitors ,
                                activation = activation_type, prior_mean=0, prior_scale=1. , kl_w=0.1, ibp = ibp)
         
        else:
            self.ln = nn.Linear(self.input_size, self.embedding_dim)
      
        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if bayesian:
            self.activation_type = False
        else:
            self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)
            
        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)
       
        if freeze:
            freeze_params(self)


    # pylint: disable=arguments-differ
    def forward_(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """

        #x = self.ln(x.transpose(-1,-2)).transpose(-1,-2)
        x=self.ln(x)
       
        if self.norm_type:
            x = self.norm(x, mask)
        
        if  self.activation_type and (not self.bayesian):
            x = self.activation(x)
        
        if self.scale:
            return x * self.scale_factor
        else:
            return x
        
    # pylint: disable=arguments-differ
    def forward( self, x: Tensor, mask: Tensor) -> Tensor:
        if self.training :
            return self.forward_(x,mask)
        else:
            
            out=[]
            #for i in range(self.inference_sample_size):
            for i in range(self.inference_sample_size):
               
               x_=  self.forward_(x,mask)
               
               out.append(torch.unsqueeze(x_,-1))
        out=torch.cat(out,-1)
      
        
        
        return out
    
    def __repr__(self):
        return "%s(embedding_dim=%d, input_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_size,
        )

class GCNSpatialEmbedding(nn.Module):
    def __init__(
            self, 
            in_channels=2, 
            hidden_dim=3, 
            embedding_dim=512, 
            num_layers=3,
            freeze: bool = False,
            **kwargs
            ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.activation = nn.ReLU()

        self.projection = nn.Linear(102 * hidden_dim, embedding_dim)
        self.batchnorm = nn.BatchNorm1d(embedding_dim)

        if freeze:
            freeze_params(self)

    def forward(self, x, mask):  
        # x: (B, T, 102, 2)
        # mask: (B, 1, T) -> convert to (B, T)
        B, T, V, C = x.shape
        mask = mask.squeeze(1)  # (B, T)

        x = x.view(B * T, V, C)
        mask_flat = mask.view(-1)  # (B*T,)

        edge_index = self.edge_index.to(x.device)

        valid_indices = mask_flat.nonzero(as_tuple=False).squeeze(1)
        valid_data = x[valid_indices]

        data_list = []
        for i, idx in enumerate(valid_indices):
            data = Data(x=valid_data[i], edge_index=edge_index)
            data.batch = torch.full((V,), i, dtype=torch.long)
            data_list.append(data)

        if len(data_list) == 0:
            # If all frames are masked, return zero tensor
            return torch.zeros(B, T, self.projection.out_features, device=x.device)

        batch = Batch.from_data_list(data_list)

        h = batch.x
        for conv in self.convs:
            h = self.activation(conv(h, batch.edge_index))  # (num_nodes, hidden_dim)

        # Reshape h: [num_nodes, hidden_dim] → [num_valid_frames, 102, hidden_dim]
        h = h.view(len(valid_indices), V, -1)

        # Flatten per-frame node features: [valid_frames, 102 * hidden_dim]
        h_flat = h.view(len(valid_indices), -1)

        # Project to final dimension
        projected = self.projection(h_flat)  # (valid_frames, embedding_dim)
        projected = self.batchnorm(projected)
        
        # Reconstruct full sequence
        output = torch.zeros(B * T, projected.size(-1), device=x.device)
        output[valid_indices] = projected
        output = output.view(B, T, -1)

        return output

    @property
    def edge_index(self):
        edges = [[0, 6], [0, 5], [6, 8], [5, 7], [0, 31], [31, 32], [32, 33], [28, 33],
                 [28, 29], [29, 30], [30, 31], [0, 43], [40, 41], [41, 42], [42, 43],
                 [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49], [49, 50],
                 [50, 51], [40, 51], [40, 52], [46, 56], [52, 53], [53, 54], [54, 55],
                 [55, 56], [56, 57], [57, 58], [58, 59], [52, 59], [0, 34], [34, 35],
                 [35, 36], [36, 37], [37, 38], [38, 39], [34, 39], [7, 60], [60, 61],
                 [61, 62], [62, 63], [63, 64], [60, 65], [65, 66], [66, 67], [67, 68],
                 [60, 69], [69, 70], [70, 71], [71, 72], [60, 73], [73, 74], [74, 75],
                 [75, 76], [60, 77], [77, 78], [78, 79], [79, 80], [8, 81], [81, 82],
                 [82, 83], [83, 84], [84, 85], [81, 86], [86, 87], [87, 88], [88, 89],
                 [81, 90], [90, 91], [91, 92], [92, 93], [81, 94], [94, 95], [95, 96],
                 [96, 97], [81, 98], [98, 99], [99, 100], [100, 101]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        rev = edge_index[[1, 0]]
        return torch.cat([edge_index, rev], dim=1)