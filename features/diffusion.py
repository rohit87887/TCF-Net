"""
Modality C: Propagation-Proxy via SIR Cascade Simulation (8-D)
Models the social spread of content using a discrete-time Susceptible-Infected-Recovered (SIR)
cascade simulator. Arousal parameters are derived strictly from text cues without seeing ground-truth labels,
guaranteeing zero data leakage while capturing the virality dynamics of emotive claims.
"""

import numpy as np
from typing import List, Optional
from features.stylometric import STYLE_NAMES

PROP_NAMES = [
    'peak_infected',
    'time_to_peak',
    'virality_slope',
    'cascade_breadth',
    'cascade_depth',
    'echo_chamber',
    'bot_likeness',
    'decay_rate'
]

def simulate_cascade(style_vec: List[float], rng: Optional[np.random.Generator] = None, sim_steps: int = 24) -> List[float]:
    """
    Simulates a 24-step discrete-time SIR contagion process based on stylometric arousal.
    Returns 8 cascade summary descriptors.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    excl_idx = STYLE_NAMES.index('excl_ratio')
    caps_idx = STYLE_NAMES.index('caps_word_ratio')
    sens_idx = STYLE_NAMES.index('sensational_density')

    excl = style_vec[excl_idx]
    caps = style_vec[caps_idx]
    sens = style_vec[sens_idx]

    # Content-based arousal proxy
    arous = np.clip(4.0 * excl + 3.0 * caps + 6.0 * sens, 0.0, 1.0)

    # Transmission beta and recovery gamma
    beta = np.clip(0.18 + 0.45 * arous + rng.normal(0, 0.03), 0.02, 0.95)
    gamma = np.clip(0.22 - 0.08 * arous + rng.normal(0, 0.02), 0.02, 0.90)

    N, I, R = 1.0, 0.002, 0.0
    S = N - I
    curve = []

    for _ in range(sim_steps):
        new_i = beta * S * I
        new_r = gamma * I
        S = max(S - new_i, 0.0)
        I = max(I + new_i - new_r, 0.0)
        R += new_r
        curve.append(I)

    curve = np.array(curve)
    peak = float(curve.max())
    t_peak = int(curve.argmax()) + 1
    slope = float(peak / t_peak)
    breadth = float(R + I)
    depth = float(np.log1p(peak * sim_steps))
    echo = float(np.clip(0.3 + 0.5 * arous + rng.normal(0, 0.05), 0.0, 1.0))
    bot = float(np.clip(0.1 + 0.6 * arous * rng.uniform(0.5, 1.2), 0.0, 1.0))
    tail = curve[t_peak:] if t_peak < len(curve) - 1 else curve[-2:]
    decay = float((peak - tail.min()) / (peak + 1e-8))

    return [
        peak,
        float(t_peak / sim_steps),
        slope,
        breadth,
        depth,
        echo,
        bot,
        decay
    ]

def batch_simulate_cascade(style_matrix: np.ndarray, seed: int = 42, sim_steps: int = 24) -> np.ndarray:
    """Simulates cascades for an entire batch of stylometric feature vectors."""
    rng = np.random.default_rng(seed)
    mat = np.array([simulate_cascade(v, rng, sim_steps) for v in style_matrix], dtype=np.float32)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

