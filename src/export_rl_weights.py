import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))

MODEL_PATH   = Path(__file__).parent.parent / "outputs" / "rl_agent" / "best_model.zip"
WEIGHTS_PATH = Path(__file__).parent.parent / "outputs" / "rl_agent" / "policy_weights.npz"

from stable_baselines3 import PPO
print(f"Loading {MODEL_PATH} ...")
model = PPO.load(str(MODEL_PATH))
sd = {k: v.cpu().numpy() for k, v in model.policy.state_dict().items()}
np.savez(str(WEIGHTS_PATH), **sd)
print(f"Saved {len(sd)} tensors → {WEIGHTS_PATH}")
for k, v in sd.items():
    print(f"  {k:55s} {v.shape}")
