import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    A base class for different classification heads.
    This can be extended to create various types of classification heads.
    """
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x):
        """
        This method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class MLPHead(ClassificationHead):
    def __init__(self, config):
        self.inference_sample_size = config['model']['inference_sample_size']
        config = config['model']['classification_head']
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        num_classes = config['num_classes']
        dropout_rate = config.get('dropout_rate', 0.2)  # Default dropout rate
        
        super(MLPHead, self).__init__(input_size, num_classes)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        if self.training:
            return self.forward_(x)
        else:
            inference_sample_size = max(x.shape[-1], self.inference_sample_size)

            out = torch.zeros((x.shape[0], self.num_classes),  device=x.device)

            for i in range(inference_sample_size):
                xi = x[..., i % x.shape[-1]]
                xi = self.forward_(xi)
                out += xi

            # Average the outputs over the number of samples
            return out*1.0/inference_sample_size

    def forward_(self, x):
        return self.network(x).mean(dim=1)


class AttentionHead(ClassificationHead):
    def __init__(self, config):
        self.inference_sample_size = config['model']['inference_sample_size']
        config = config['model']['classification_head']
        self.input_size = config['input_size']
        self.num_classes = config['num_classes']
        
        super(AttentionHead, self).__init__(self.input_size, self.num_classes)
        self.attention = nn.Linear(self.input_size, 1)
        self.norm = nn.LayerNorm(self.input_size)
        self.fc_1 = nn.Linear(self.input_size, self.num_classes*2)
        self.fc_2 = nn.Linear(self.num_classes*2, self.num_classes)

    def forward(self, x):
        if self.training:
            return self.forward_(x)
        else:
            inference_sample_size = max(x.shape[-1], self.inference_sample_size)

            out = torch.zeros((x.shape[0], self.num_classes),  device=x.device)
            emb = torch.zeros((x.shape[0], self.input_size),  device=x.device)

            for i in range(inference_sample_size):
                xi = x[..., i % x.shape[-1]]
                xi = self.forward_(xi)
                out += xi[0]
                emb += xi[1]

            # Average the outputs over the number of samples
            return out*1.0/inference_sample_size, emb*1.0/inference_sample_size


    def forward_(self, x):
        # Compute attention weights
        weights = torch.softmax(self.attention(x), dim=1)  # (batch, seq, 1)
        x = (x * weights).sum(dim=1)  # Weighted sum across the sequence
        x = self.norm(x)
        return self.fc_2(self.fc_1(x)), x


class ConvHead(ClassificationHead):
    def __init__(self, config):
        self.inference_sample_size = config['model']['inference_sample_size']
        config = config['model']['classification_head']
        self.input_size = config['input_size']
        self.num_classes = config['num_classes']
        self.kernel_size = config.get('kernel_size', 3)  # Default kernel size
        self.num_filters = config.get('num_filters', 128)  # Default number of filters

        
        super(ConvHead, self).__init__(self.input_size, self.num_classes)
        self.network = nn.Sequential(
            nn.Conv1d(self.input_size, self.num_filters, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm1d(self.num_filters),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pooling over the sequence length
            nn.Flatten(),
            nn.Linear(self.num_filters, self.num_classes)
        )

    def forward(self, x):
        if self.training:
            # print(x.shape)
            x = x.permute(0, 2, 1)  # Reshape to (batch, input_size, seq_length)
            return self.network(x)
        else:
            inference_sample_size = max(x.shape[-1], self.inference_sample_size)

            out = torch.zeros((x.shape[0], self.num_classes),  device=x.device)

            for i in range(inference_sample_size):
                xi = x[..., i % x.shape[-1]]
                xi = self.network(xi.permute(0, 2, 1) )
                out += xi

            # Average the outputs over the number of samples
            out = out*1.0/inference_sample_size
            return out


class RNNHead(ClassificationHead):
    def __init__(self, config):
        self.inference_sample_size = config['model']['inference_sample_size']
        config = config['model']['classification_head']
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        num_classes = config['num_classes']
        
        super(RNNHead, self).__init__(input_size, num_classes)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        if self.training:
            return self.forward_(x)
        else:
            inference_sample_size = max(x.shape[-1], self.inference_sample_size)

            out = torch.zeros((x.shape[0], self.num_classes),  device=x.device)

            for i in range(inference_sample_size):
                xi = x[..., i % x.shape[-1]]
                xi = self.forward_(xi)
                out += xi

            # Average the outputs over the number of samples
            return out*1.0/inference_sample_size


    def forward_(self, x):
        _, (h_n, _) = self.rnn(x)  # Get the final hidden state
        h_n = self.norm(h_n.squeeze(0))
        return self.fc(h_n)  # (batch_size, num_classes)

