import pytest
import torch
from transformers import GPT2Config
from gpt2 import *


def test_gpt2config():
    configuration = GPT2Config()
    assert configuration.vocab_size == 50257
    assert configuration.n_positions == 1024

def test_mlp():
    configuration = GPT2Config(n_embd=5)
    if configuration.n_inner is None:
        configuration.n_inner = 4 * configuration.n_embd
    mlp = MLP(configuration)
    assert mlp(torch.ones(configuration.n_embd)).shape[0] == configuration.n_embd


# test_mlp()
