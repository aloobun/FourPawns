"""
mcts??
treenode for representing board position. what it tracks? initial guess, win loss score,
track of how many times it has visited state and avg score may be to tell how good is the state
"""

import numpy as np
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

class TreeNode:
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and visit count N."""
    def __init__(self, parent: Optional['TreeNode'], prior_p: float):
        self._parent = parent
        self._children: Dict[int, TreeNode] = {}
        self._visit_count = 0
        self._total_action_value = 0.0
        self._mean_action_value = 0.0
        self._prior_probability = prior_p

    def expand(self, action_priors: np.ndarray):
        for action, prob in enumerate(action_priors):
            if prob > 0 and action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct: float) -> Tuple[int, 'TreeNode']:
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update_recursive(self, leaf_value: float):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def update(self, leaf_value: float):
        self._visit_count += 1
        self._total_action_value += leaf_value
        self._mean_action_value = self._total_action_value / self._visit_count

    def get_value(self, c_puct: float) -> float:
        if self._parent is None:
            parent_visit_count = 1
        else:
            parent_visit_count = self._parent.visit_count

        u = c_puct * self._prior_probability * math.sqrt(parent_visit_count) / (1 + self._visit_count)
        q = self._mean_action_value
        return q + u

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def visit_count(self) -> int:
        return self._visit_count
        
    @property
    def children(self) -> Dict[int, 'TreeNode']:
        return self._children


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, num_simulations=800):
        self._root = TreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._num_simulations = num_simulations

    def _playout(self, env_state):
        node = self._root
        
        while not node.is_leaf:
            action, node = node.select(self._c_puct)
        action_probs, leaf_value = self._policy_value_fn(env_state)
        node.expand(action_probs)
        node.update_recursive(-leaf_value)

    def get_action_probabilities(self, env_state, temp=1e-3) -> np.ndarray:
        for n in range(self._num_simulations):
            self._playout(env_state)
            
        # calculate the final move probabilities based on visit counts at the root.
        visit_counts = np.array([
            node.visit_count for action, node in sorted(self._root.children.items())
        ])
        
        if temp == 0:
            # choose the move with the highest visit count
            action_probs = np.zeros_like(visit_counts, dtype=np.float32)
            best_action_idx = np.argmax(visit_counts)
            action_probs[best_action_idx] = 1.0
        else:
            # sample from a distribution proportional to N^(1/temp)
            action_probs = visit_counts**(1/temp)
            action_probs /= np.sum(action_probs)
            
        return action_probs

    def update_with_move(self, last_move: int):
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root._parent = None # Detach from the old parent
        else:
            self._root = TreeNode(None, 1.0)


def run_mcts_unit_test():
    TEST_CONFIG = {
        "board_shape": (8, 8),
        "in_channels": 111,
        "action_space_size": 4672,
        "num_simulations": 16,
        "c_puct": 5,
    }


    class MockPolicyValueNet:
        def __call__(self, state_input):
            action_size = TEST_CONFIG["action_space_size"]
            legal_moves_mask = np.zeros(action_size)
            legal_moves_mask[:20] = 1
            

            action_probs = legal_moves_mask / np.sum(legal_moves_mask)
            
            value = 0.0
            return action_probs, value

    dummy_game_state = "start_position"
    
    print("Instantiating MCTS with a dummy network")
    mock_net = MockPolicyValueNet()
    mcts_instance = MCTS(mock_net,
                         c_puct=TEST_CONFIG["c_puct"],
                         num_simulations=TEST_CONFIG["num_simulations"])

    print(f"Running {TEST_CONFIG['num_simulations']} simulations...")
    action_probabilities = mcts_instance.get_action_probabilities(dummy_game_state, temp=1.0)
    print("MCTS run completed.")

    print("\n--- Verifying MCTS Output ---")
    print(f"Output action probabilities shape: {action_probabilities.shape}")
    assert action_probabilities.shape == (20,), "Probabilities should only cover expanded child nodes"

    print(f"Sum of probabilities: {np.sum(action_probabilities):.6f}")
    assert np.isclose(np.sum(action_probabilities), 1.0), "Probabilities must sum to 1."
    
    print(f"Number of non-zero probabilities: {np.count_nonzero(action_probabilities)}")
    
    is_uniform = np.allclose(action_probabilities, action_probabilities[0])
    assert not is_uniform, "Probabilities should not be uniform after MCTS search."
    print("  - Output is a valid, non-uniform probability distribution.")

    print("\n*************")
    print("MCTS Unit Test Passed ")
    print("*************")


if __name__ == "__main__":
    run_mcts_unit_test()
