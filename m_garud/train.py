
import torch
import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from safetensors.torch import save_file

from m_garud.config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LOG_DIR,
    LANGUAGE_MAP, HIDDEN_DIM, NUM_HEADS, NUM_KV_GROUPS,
    MBERT_PATH, PATIENCE
)

from m_garud.data.alternating_loader import AlternatingTribalLoader
from m_garud.models.build import build_models
from m_garud.training.trainer import MGARuDTrainer
from m_garud.eval import evaluate_epoch

def train():
    print(f"🚀 Initializing M-GARuD Training on: {DEVICE}")
    tribal_langs = list(LANGUAGE_MAP.keys())
    
    # --- State Tracking ---
    patience_limit = PATIENCE
    early_stop_counter = 0         
    best_macro_task_loss = float("inf") # Track raw performance, not total loss

    student, pivot, fusion, teacher_encoders = build_models()
    train_loader = AlternatingTribalLoader(tribal_langs, BATCH_SIZE, split="train")
    val_loader = AlternatingTribalLoader(tribal_langs, BATCH_SIZE, split="val")

    trainer = MGARuDTrainer(
        student=student, pivot=pivot, fusion=fusion, 
        teacher_encoders=teacher_encoders, dataloader=train_loader, device=DEVICE
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history = defaultdict(list)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        
        # --- SAFE MULTITASK TRIGGER (Delayed Freezing) ---
        # Stop optimizing uncertainty once tasks have likely stabilized
        if epoch == 20:
            print(f"❄️ Epoch {epoch}: Freezing Uncertainty Weights to prevent variance collapse.")
            trainer.uw_loss.log_var_span.requires_grad = False
            trainer.uw_loss.log_var_nt.requires_grad = False
        
        epoch_losses = defaultdict(list)
        lang_step_counts = defaultdict(int) 
        
        trainer.student.train()
        trainer.pivot.train()
        trainer.fusion.train()

        steps_per_epoch = min(len(ds)//BATCH_SIZE for ds in train_loader.datasets.values()) * len(tribal_langs)

        for _ in tqdm(range(steps_per_epoch), desc="Training"):
            try:
                batch = next(train_loader)
                lang = batch.get('tribal_lang', 'unknown')
                lang_step_counts[lang] += 1
                
                losses = trainer.train_step(batch)
                for k, v in losses.items():
                    epoch_losses[k].append(v)
            except StopIteration:
                break

        train_metrics = {f"train_{k}": sum(v) / len(v) for k, v in epoch_losses.items()}

        # Uncertainty Inspection
        span_uw = trainer.uw_loss.log_var_span.item()
        nt_uw = trainer.uw_loss.log_var_nt.item()
        print(f"🧐 Uncertainty: span = {span_uw:.18f} | nt = {nt_uw:.18f}")

        # --- VALIDATION ---
        val_metrics = evaluate_epoch(trainer=trainer, dataloader=val_loader)

        # Calculate Macro-Task Loss (RAW Task Losses only, no uncertainty weights)
        # This is the best suited metric for imbalanced distillation
        lang_raw_losses = [val_metrics[f"val_loss_{l}"] for l in tribal_langs if f"val_loss_{l}" in val_metrics]
        current_macro_task_loss = sum(lang_raw_losses) / len(lang_raw_losses) if lang_raw_losses else val_metrics.get("val_total", float("inf"))

        print(f"📉 Train Metrics: {train_metrics}")
        print(f"📊 Macro-Task Val Loss: {current_macro_task_loss:.4f} (Criterion for Best Model)")
        for lang in tribal_langs:
            print(f"  - {lang} val loss: {val_metrics.get(f'val_loss_{lang}', 'N/A')}")
        
        # History persistence
        for k, v in train_metrics.items(): history[k].append(v)
        for k, v in val_metrics.items(): history[k].append(v)
        history["macro_task_loss"].append(current_macro_task_loss)
        history["span_uw"].append(span_uw)
        history["nt_uw"].append(nt_uw)

        # --- CHECKPOINTING (Based on Semantic/Linguistic Task Loss) ---
        if current_macro_task_loss < best_macro_task_loss:
            best_macro_task_loss = current_macro_task_loss
            early_stop_counter = 0  
            
            torch.save({
                "epoch": epoch,
                "student_state": trainer.student.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "uw_state": trainer.uw_loss.state_dict(),
                "best_macro_task_loss": best_macro_task_loss,
            }, LOG_DIR / "checkpoint_full.pt")

            model_dir = LOG_DIR / "final_student_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            trainable_weights = {k: v for k, v in trainer.student.state_dict().items() if "mbert" not in k}
            save_file(trainable_weights, model_dir / "model.safetensors")
            
            with open(model_dir / "config.json", "w") as f:
                json.dump({
                    "num_layers": 6, "hidden_dim": HIDDEN_DIM, 
                    "num_heads": NUM_HEADS, "num_kv_groups": NUM_KV_GROUPS,
                    "mbert_path": str(MBERT_PATH), "languages": tribal_langs
                }, f, indent=2)
            
            print(f"✨ Task-based improvement! Model saved (Task Loss: {current_macro_task_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"🐢 No task improvement for {early_stop_counter}/{patience_limit} epochs.")

        if early_stop_counter >= patience_limit:
            print(f"🛑 Converged. Stopping at Epoch {epoch}.")
            break

        with open(LOG_DIR / "loss_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n✅ Training Complete.")

if __name__ == "__main__":
    train()