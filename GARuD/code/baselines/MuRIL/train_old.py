import os 
import gc
import numpy as np
import pandas as pd 
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import logging
from datetime import datetime
from config import *  # Assuming MODEL_CARD, OUTPUT_DIR, LOGGING_DIR, etc. are defined here


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(LOGGING_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_CARD).to(DEVICE)


    train_df = pd.read_csv("/teamspace/studios/this_studio/train_90p.csv")
    val_df = pd.read_csv("/teamspace/studios/this_studio/val_10p.csv")
    train_df = train_df.drop(columns=["Marathi", "Gujarati"])
    train_df = train_df.dropna(subset=["Hindi", "Bhili"]).reset_index(drop=True)
    train_df[train_df["Hindi"].str.strip().astype(bool)].reset_index(drop=True)
    bhili_texts_train = train_df["Bhili"].tolist()  # Limit to 10,000 samples for training

    val_df = val_df.drop(columns=["Marathi", "Gujarati"])
    val_df = val_df.dropna(subset=["Hindi", "Bhili"]).reset_index(drop=True)
    val_df = val_df[val_df["Hindi"].str.strip().astype(bool)].reset_index(drop=True)
    bhili_texts_val = val_df["Bhili"].tolist()

    del train_df, val_df
    gc.collect()

    # Create Hugging Face Dataset objects from lists of texts
    raw_datasets = DatasetDict({
        "train": Dataset.from_dict({"text": bhili_texts_train}),
        "validation": Dataset.from_dict({"text": bhili_texts_val})
    })

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            padding='longest',
            truncation=True,
            max_length=128
            )

    # Tokenize the datasets
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(), # Use multiple processes for faster tokenization
        remove_columns=["text"]  # Remove the original text column to save memory
    )

    del raw_datasets
    gc.collect()
 
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,              # Enable Masked Language Modeling
        mlm_probability=MLM_PROBABILITY, # Probability of masking a token
        return_tensors="pt"  
    )

    num_train_samples = len(tokenized_datasets["train"])
    # Calculate steps per epoch considering gradient accumulation
    steps_per_epoch = num_train_samples / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    total_training_steps = int(steps_per_epoch * NUM_TRAIN_EPOCHS)
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    logger.info("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,              # Overwrite the output directory if it already exists
        num_train_epochs=NUM_TRAIN_EPOCHS,      # Total number of training epochs
        per_device_train_batch_size=BATCH_SIZE, # Batch size per device during training
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Number of steps to accumulate gradients
        per_device_eval_batch_size=4,           # Batch size per device during evaluation
        eval_accumulation_steps=2,  # Accumulate gradients during evaluation
        gradient_checkpointing=True,
        eval_strategy="epoch",                  # Evaluate at the end of each epoch
        save_strategy="epoch",                  # Save checkpoint at the end of each epoch
        logging_dir=LOGGING_DIR,                # Directory for storing logs
        logging_steps=100,                      # Log every X update steps
        learning_rate=LEARNING_RATE,            # Initial learning rate
        weight_decay=0.01,                      # Strength of weight decay
        fp16=torch.cuda.is_available(),         # Use mixed precision training if GPU is available
        load_best_model_at_end=True,            # Load the best model (based on eval_loss) at the end of training
        metric_for_best_model="eval_loss",      # Metric to use for comparison
        greater_is_better=False,                # For loss, lower is better
        report_to="tensorboard",                # Enable TensorBoard logging
        push_to_hub=False,                      # Set to True to push the model to Hugging Face Hub (requires `huggingface-cli login`)
        lr_scheduler_type="linear",             # Use a linear learning rate scheduler
        warmup_steps=warmup_steps,              # Number of warmup steps
        logging_strategy="steps",                           # Enable verbose logging
    )
    logger.info("TrainingArguments set up.")

    def compute_metrics(eval_pred):
        """Computes MLM accuracy and returns it along with loss."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # Ignore masked tokens (where labels != -100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    logger.info("Initializing Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,  # Pass tokenizer to Trainer for consistency and potential special token handling
        data_collator=data_collator,
        # compute_metrics=compute_metrics, # Add the compute_metrics function
    )
    logger.info("Trainer initialized successfully.")

    logger.info("Starting model training. This will save checkpoints periodically.")
    try:
        train_result = trainer.train()
        logger.info("Training complete.")
        # Log training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)  # Save metrics to file
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")

    logger.info("Evaluating the trained model on the validation set...")
    try:
        eval_results = trainer.evaluate()
        # Calculate perplexity: exp(loss)
        perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
        logger.info(f"Evaluation results: {eval_results}")
        logger.info(f"Perplexity: {perplexity}")
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results) # Save metrics to file
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")


    # The Trainer automatically loads the best model when `load_best_model_at_end=True`.
    # We save this best model and its corresponding tokenizer.
    logger.info(f"Saving the best model and tokenizer to {OUTPUT_DIR}...")
    try:
        trainer.save_model(OUTPUT_DIR) # Saves the best model found during training
        tokenizer.save_pretrained(OUTPUT_DIR) # Saves the tokenizer used
        logger.info("Best model and tokenizer saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model or tokenizer: {e}")

    logger.info("Fine-tuning/pre-training process completed.")


if __name__ == "__main__":
    main()