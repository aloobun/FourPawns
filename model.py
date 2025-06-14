"""
after so many questions with opus, gemini and chatgpt ive decided to use a hybrid network,
cnn + lm policy/value network.
model should be able to take a batch of obervations and give policy and value output.

1. may be like a conv layers (3-5) with relu and batch norm - input is (batch_size, 111,8,8))
2. and then load a smol llm and for now ill freeze weights (as i dont want to train it).
3. a linear layer should map the cnn output to llm hidden dim.
4. observation -> cnn -> feature vector -> bridge from cnn to llm -> llm embedding (and pass this though llm model)
5. Policy: ?? still trying to figure out, may be a linear layer to final hidden state that maps. 
6. yolo
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

LLAMA_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

class ResBlock(nn.module):
  #forward()
  return

class PolicyValueNet(nn.module):
  #forward()
  return
