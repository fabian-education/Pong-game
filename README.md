# Pong (Human + Deep Q-Learning)

Small Pong project with:
- a classic human-vs-human Pong (`pong_human_original.py`)
- a simple RL environment (`pong_rl_environment.py`)
- Deep Q-Learning agents (`deepQ_agent.py`) and example training scripts

## Requirements
- Python >= 3.13
- Dependencies: `pygame`, `numpy`, `tensorflow`

## Run
Human Pong (W/S vs ↑/↓):
```bash
python pong_human_original.py
```

Train/play single agent (right paddle):
```bash
python example_w_agent.py
```

Train/play two agents (left + right):
```bash
python example_two_agents.py
```

## Notes
- The RL state is 6 values: ball position/velocity (normalized) + both paddle y positions.
- Models are saved/loaded as `.keras` files (e.g. `pong_single_agent.keras`, `pong_left.keras`, `pong_right.keras`).

## Todo
- Punish agent when goal was scored and he was just sitting in the corner
