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
MODEL_DTYPE = torch.bfloat16

#residual block, 2 conv layers
class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

#policy value network
class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_shape: Tuple[int, int],
        in_channels: int,
        action_space_size: int,
        cnn_channels: int = 256,
        num_res_blocks: int = 5,
        llama_model_name: str = LLAMA_MODEL_NAME,
        model_dtype: torch.dtype = MODEL_DTYPE,
    ):
        super(PolicyValueNet, self).__init__()
        print("Init")
        self.model_dtype = model_dtype

        self.board_shape = board_shape
        self.in_channels = in_channels
        self.action_space_size = action_space_size

        self.cnn_head = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            *[ResBlock(cnn_channels) for _ in range(num_res_blocks)]
        )
        self.cnn_output_size = cnn_channels * board_shape[0] * board_shape[1]
        print(f" Output features: {self.cnn_output_size}")

        print(f" Loading llm: {llama_model_name}...")
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_name,
            torch_dtype=self.model_dtype,
        )
        self.llama_hidden_size = self.llama_model.config.hidden_size
        print(f" LLM loaded. Hidden size: {self.llama_hidden_size}")

        for param in self.llama_model.parameters():
            param.requires_grad = False
        print(" All LLM weights have been frozen.")

        # like a bridge from cnn to llm
        self.bridge = nn.Linear(self.cnn_output_size, self.llama_hidden_size)
        print(" Bridge layer created.")

        self.policy_tail = nn.Linear(self.llama_hidden_size, action_space_size)
        self.value_tail = nn.Linear(self.llama_hidden_size, 1)
        print(" Policy and Value 'tails' created.")
        print("Model initialization complete.")

        print(f" Converting custom layers to {self.model_dtype}")
        self.cnn_head.to(self.model_dtype)
        self.bridge.to(self.model_dtype)
        self.policy_tail.to(self.model_dtype)
        self.value_tail.to(self.model_dtype)
        print("Model init complete.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: x - A batch of board states, shape (N, C, H, W)
        Returns: A tuple of (policy_logits, value)
        """
        x = x.to(self.model_dtype)
        # pass through cnn head
        cnn_out = self.cnn_head(x)
        flattened = cnn_out.view(-1, self.cnn_output_size)
        
        # project through the Bridge
        board_embedding = self.bridge(flattened)
        
        # pass through the llm
        board_embedding = board_embedding.unsqueeze(1)
        llama_outputs = self.llama_model.model(inputs_embeds=board_embedding)
        final_hidden_state = llama_outputs.last_hidden_state
        final_feature_vector = final_hidden_state.squeeze(1)

        # get predictions from the output tails
        policy_logits = self.policy_tail(final_feature_vector)
        value_logit = self.value_tail(final_feature_vector)
        value = torch.tanh(value_logit)
        
        return policy_logits, value

def run_unit_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    TEST_CONFIG = {
        "board_shape": (8, 8),
        "in_channels": 111,
        "action_space_size": 4672,
        "batch_size": 4
    }

    try:
        print("\nInstantiating model with test configuration...")
        model = PolicyValueNet(
            board_shape=TEST_CONFIG["board_shape"],
            in_channels=TEST_CONFIG["in_channels"],
            action_space_size=TEST_CONFIG["action_space_size"]
        ).to(device)
        model.eval()

        dummy_input_shape = (
            TEST_CONFIG["batch_size"],
            TEST_CONFIG["in_channels"],
            TEST_CONFIG["board_shape"][0],
            TEST_CONFIG["board_shape"][1]
        )
        print(f"\nCreating a dummy input tensor of shape: {dummy_input_shape}")
        dummy_input = torch.randn(dummy_input_shape).to(device)

        print("Performing a forward pass through the network...")
        with torch.no_grad():
            policy_logits, value = model(dummy_input)
        print("Forward pass successful.")

        print("\n--- Verifying Output Shapes ---")
        expected_policy_shape = (TEST_CONFIG["batch_size"], TEST_CONFIG["action_space_size"])
        expected_value_shape = (TEST_CONFIG["batch_size"], 1)

        print(f"Policy Logits Shape: {policy_logits.shape} (Expected: {expected_policy_shape})")
        assert policy_logits.shape == expected_policy_shape, "Policy output shape is incorrect!"
        assert policy_logits.dtype == MODEL_DTYPE
        print("  - Policy shape: Correct.")

        print(f"Value Shape: {value.shape} (Expected: {expected_value_shape})")
        assert value.shape == expected_value_shape, "Value output shape is incorrect!"
        print("  - Value shape: Correct.")
        
        print("\n--- Verifying Trainable Parameters ---")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        print(f"Total Trainable Parameters: {trainable_params:,}")
        print(f"Total Frozen Parameters:    {frozen_params:,}")
        assert trainable_params > 0, "No parameters are trainable!"
        assert frozen_params > 0, "No parameters are frozen!"
        print("  - Parameter freezing setup is correct.")

        print("\n=============================================")
        print("         Model Unit Test Passed!           ")
        print("=============================================")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("          Model Unit Test FAILED!            ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    run_unit_test()
