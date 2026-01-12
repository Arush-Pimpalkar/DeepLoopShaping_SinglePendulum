"""RL_Arush - Deep Loop Shaping for Pendulum Control."""

from pathlib import Path

# Project root directory (goes up from src/rl_arush to RL_Arush)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HDF5_DIR = DATA_DIR / "hdf5"
NOISE_INPUTS_DIR = DATA_DIR / "noise_inputs"
TRANSFER_FUNCTIONS_DIR = DATA_DIR / "transfer_functions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = Path(__file__).parent / "models"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR", 
    "HDF5_DIR",
    "NOISE_INPUTS_DIR",
    "TRANSFER_FUNCTIONS_DIR",
    "OUTPUTS_DIR",
    "MODELS_DIR",
]
