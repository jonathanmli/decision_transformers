import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import GPT2Config
from transformers.activations import ACT2FN

class Decoder(nn.Module):
    '''
    Stack N decoder layers
    '''

    def __init__(self, config) -> None:
        '''
        Config should contain the parameters as described in huggingface.GPT2Config
        '''
        super().__init__()
        self.config = config
        self.vocab_size
        self.n_positions
        self.n_embd
        self.n_layer
        self.n_head
        self.n_inner
        self.activation_function
        self.resid_pdrop
        self.attn_pdrop
        self.layer_norm_epsilon

class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = MLP()
        self.attention = Attention()

    def forward(self, x):
        # Masked self attention
        x = self.attention(x)

        # feed forward
        x = self.mlp(x)
        return x

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_inner)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()

    def _attention(self, q, k, v, m=None):
        pass

    def forward(self, x):
        return x


