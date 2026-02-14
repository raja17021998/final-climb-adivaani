import os 
import torch

MODEL_CARD = "google/muril-base-cased"  # Base MuRIL model to start with
OUTPUT_DIR = "./checkpoints"         # Directory to save the fine-tuned model and tokenizer
LOGGING_DIR = "./logs"               # Directory for training logs
MAX_SEQ_LEN = 128                    # Maximum sequence length for tokenization
MLM_PROB = 0.15                       # Probability of masking tokens for MLM task (standard is 0.15)
EPOCHS = 20                         # Number of training epochs
BATCH_SIZE = 8                              # Batch size per GPU/CPU
GRADIENT_ACCUMULATION_STEPS = 2              # Number of gradient accumulation steps
LR = 5e-5                         # Learning rate for the optimizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use for training (GPU or CPU)