import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
BATCH_SIZE = 100
NUMBER_OF_TRIALS = 3
