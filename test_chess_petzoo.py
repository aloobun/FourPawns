"""
init new chess environment and play one game with random policies.
why random? becuase its weekend (15 june 2025) and i can do whatever i want becuae im fucking free from corporate bullshit.
i want to verify env interaction loop and ispect observation and action mark (refer to docs to know what they are).

Expected Observation Tensor: (8, 8, 111), bool
Expected Action Mask: (4672,), int8
"""

from string import printable
import numpy as np
from pettingzoo.classic import chess_v6
from pettingzoo.utils.env import AECEnv

def describe_observation(observation: dict):
    """details about the observation tensor"""
    agent_name = observation.get("agent_name_for_debug", "current agent")
    obs_tensor = observation["observation"]
    action_mask = observation["action_mask"]

    print(f"\n Obseervation for {agent_name}")
    print(f" Observation tensor: {obs_tensor.shape}")
    print(f" Observation tensor dtype: {obs_tensor.dtype}")

    #large tensor perhaps so we check min/max
    print(f"min/max: {obs_tensor.min()/obs_tensor.max()}")

    print(f" Action mask tensor: {action_mask.shape}")
    print(f" action mask dtype: {action_mask.dtype}")

    #0 for illegal move and 1 for legal move
    num_legal_moves = np.sum(action_mask)
    print(f"number of legal moces : {num_legal_moves}")
    print("-" * (len(agent_name) + 26))

def play_random_chess_game():
    """
    init the chess_v6 env and plays one game from start to finish
    """
    try:
        env: AECEnv = chess_v6.env(render_mode="ansi")
        print("Env init successfully.")
    except Exception as e:
        print(f"Error initializing env: {e}")
        return

    print("\nResetting the env for a new game (seed=42)...")
    env.reset(seed=42)
    print("First board state:")
    print(env.render())

    move_count = 0
    checked_agents = set()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if reward != 0:
                outcome = "won" if reward > 0 else "lost"
                print(f"\nGame Over")
                print(f"Agent {agent} has {outcome}. Final reward: {reward}")
        else:
            assert observation is not None, "should always have a valid observation."

            if agent not in checked_agents:
                print(f"\nfirst observation check for '{agent}'")
                # Add agent name to dict for cleaner logging
                observation["agent_name_for_debug"] = agent
                describe_observation(observation)
                checked_agents.add(agent)

            legal_moves_indices = np.where(observation['action_mask'] == 1)[0]
            action = np.random.choice(legal_moves_indices)

        env.step(action)

        if not (termination or truncation):
            move_count += 1
            if move_count % 10 == 0:
                print(f"\n--- After move {move_count} (played by {agent}) ---")
                print(env.render())

    env.close()
    print(f"The random game lasted for {move_count} moves.")

if __name__ == "__main__":
    play_random_chess_game()
