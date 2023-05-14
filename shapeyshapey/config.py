import torch

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001

# Dataset
DATA_DIR = './dataset'
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = 4
PRECISION = 16 if torch.cuda.is_available() else 32
