"""
Gymnasium environment for single-home battery dispatch.

State  : [soc_norm, price_now, price_1h, price_2h, price_3h,
          load_now, load_1h, load_2h, hour_sin, hour_cos]
Action : continuous in [-1, 1]
         +1 → charge at max rate,  -1 → discharge at max rate
Reward : −(grid_cost_this_hour)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass


@dataclass
class EnvConfig:
    capacity_kwh: float = 10.0
    max_charge_rate_kw: float = 2.5
    max_discharge_rate_kw: float = 2.5
    initial_soc: float = 0.2
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    lookahead: int = 3          # hours of price/load forecast visible to agent


class BatteryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: np.ndarray, loads: np.ndarray, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.prices = np.asarray(prices, dtype=np.float32)
        self.loads  = np.asarray(loads,  dtype=np.float32)
        assert len(self.prices) == len(self.loads), "prices and loads must be same length"

        k = self.cfg.lookahead
        # obs: soc_norm + (k+1) prices + (k+1) loads + hour_sin + hour_cos
        obs_dim = 1 + (k + 1) + (k + 1) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._t = 0
        self._soc = self.cfg.capacity_kwh * self.cfg.initial_soc

    # ── helpers ──────────────────────────────────────────────────────────────

    def _obs(self):
        k   = self.cfg.lookahead
        t   = self._t
        n   = len(self.prices)
        soc_norm = self._soc / self.cfg.capacity_kwh

        price_window = [self.prices[min(t + i, n - 1)] for i in range(k + 1)]
        load_window  = [self.loads[min(t + i, n - 1)]  for i in range(k + 1)]

        hour = t % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # normalise prices and loads to roughly [0, 1]
        p_norm = np.array(price_window, dtype=np.float32) / 0.35
        l_norm = np.array(load_window,  dtype=np.float32) / 5.0

        return np.array(
            [soc_norm] + p_norm.tolist() + l_norm.tolist() + [hour_sin, hour_cos],
            dtype=np.float32,
        )

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # random start position in first 80 % of data so episode fits
        max_start = max(0, len(self.prices) - 24 * 7)
        self._t   = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._soc = self.cfg.capacity_kwh * self.cfg.initial_soc
        return self._obs(), {}

    def step(self, action):
        cfg  = self.cfg
        t    = self._t
        load = float(self.loads[t])
        price= float(self.prices[t])
        a    = float(np.clip(action[0], -1.0, 1.0))

        if a >= 0:
            # charge
            desired = a * cfg.max_charge_rate_kw
            actual  = min(desired,
                          (cfg.capacity_kwh - self._soc) / cfg.charge_efficiency)
            actual  = max(actual, 0.0)
            self._soc += actual * cfg.charge_efficiency
            charge   = actual
        else:
            # discharge
            desired  = -a * cfg.max_discharge_rate_kw
            actual   = min(desired,
                           self._soc * cfg.discharge_efficiency,
                           max(load, 0.0))
            actual   = max(actual, 0.0)
            self._soc -= actual / cfg.discharge_efficiency
            charge   = -actual

        grid_draw = max(load + charge, 0.0)
        cost      = grid_draw * price
        reward    = -cost

        self._t  += 1
        done      = self._t >= len(self.prices)
        obs       = self._obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, {}

    def render(self):
        pass
