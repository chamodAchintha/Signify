import math
import torch

from torch import nn, Tensor
import torch.nn.functional as F
from signjoey.helpers import freeze_params
from signjoey.layers import  DenseBayesian,EmbeddingBayesian
from torch_geometric.nn import GCNConv  # Import GCN layer


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





class GNNEmbeddings(nn.Module):
    edges = [[0, 6], [0, 5], [6, 8], [5, 7], [0, 62], [62, 63], [63, 64], [59, 64], [59, 60], [60, 61], [61, 62], [0, 74], [71, 72], [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [71, 82], [71, 83], [77, 87], [83, 84], [84, 85], [85, 86], [86, 87], [87, 88], [88, 89], [89, 90], [83, 90], [0, 65], [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [65, 70], [7, 91], [91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100], [100, 101], [101, 102], [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110], [110, 111], [8, 112], [112, 113], [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121], [121, 122], [122, 123], [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131], [131, 132]]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    def __init__(
        self,
        embedding_dim: int,
        input_size: int,
        gnn_hidden_dim: int,
        edge_index: torch.Tensor = edge_index,
        freeze: bool = False,
        activation_type: str = 'relu',
        scale: bool = False,
        scale_factor: float = None,
    ):
        """
        Create embeddings using GNN layers.

        :param embedding_dim: Dimension of the output embeddings.
        :param input_size: Dimension of the input features (should be 2 for [x, y]).
        :param gnn_hidden_dim: Hidden dimension for the GNN layer.
        :param edge_index: Edge index defining the graph structure.
        :param freeze: Freeze the embeddings during training.
        :param activation_type: Activation function to use.
        :param scale: Whether to scale embeddings.
        :param scale_factor: Scaling factor.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size  # Should be 2 for [x, y]
        self.gnn_hidden_dim = gnn_hidden_dim

        # Define GCN layer
        self.gnn = GCNConv(input_size, gnn_hidden_dim)

        # Linear layer to project GNN output to the desired embedding_dim
        self.proj = nn.Linear(gnn_hidden_dim, embedding_dim)

        # Activation function
        self.activation = get_activation(activation_type) if activation_type else None

        # Graph structure (edge index)
        self.edge_index = edge_index

        # Scaling factor
        self.scale = scale
        if self.scale:
            self.scale_factor = scale_factor if scale_factor else math.sqrt(self.embedding_dim)

        # Freeze parameters if needed
        if freeze:
            freeze_params(self)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through GNN embedding layer.

        :param x: Input features (batch_size, 204).
        :param mask: Optional mask for inputs.
        :return: GNN-based embeddings (batch_size, 102, embedding_dim).
        """
        print(x.size())
        print("Testing")
        batch_size, _ = x.size()

        # Reshape input: (batch_size, 204) -> (batch_size, 102, 2)
        x = x.view(batch_size, 102, 2)

        # Process each graph in the batch
        outputs = []
        for i in range(batch_size):
            node_features = x[i]  # Shape: (102, 2)
            
            # Apply GNN to compute node embeddings
            gnn_output = self.gnn(node_features, self.edge_index)  # Shape: (102, gnn_hidden_dim)
            
            if self.activation:
                gnn_output = self.activation(gnn_output)

            # Project to embedding_dim
            embeddings = self.proj(gnn_output)  # Shape: (102, embedding_dim)
            outputs.append(embeddings)

        # Stack results for the batch: (batch_size, 102, embedding_dim)
        outputs = torch.stack(outputs, dim=0)

        # Optional scaling
        if self.scale:
            outputs *= self.scale_factor

        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, input_size={self.input_size}, gnn_hidden_dim={self.gnn_hidden_dim})"
