
MODEL_DIR = "../../../models"
OUTPUT_DIR = "./checkpoints"         # Directory to save the fine-tuned model and tokenizer
LOGGING_DIR = "./logs"               # Directory for training logs
MAX_SEQ_LEN = 128                    # Maximum sequence length for tokenization
MLM_PROB = 0.15                       # Probability of masking tokens for MLM task (standard is 0.15)
EPOCHS = 500                         # Number of training epochs
BATCH_SIZE = 64                              # Batch size per GPU/CPU
LR = 5e-5                         # Learning rate for the optimizer
