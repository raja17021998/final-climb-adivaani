# import json
# from pathlib import Path
# import matplotlib.pyplot as plt
# import math

# from m_garud.config import LOG_DIR


# def plot_losses():
#     history_path = LOG_DIR / "loss_history.json"
#     if not history_path.exists():
#         raise FileNotFoundError(f"{history_path} not found")

#     with open(history_path) as f:
#         history = json.load(f)

#     epochs = range(1, len(history["train_total"]) + 1)

#     # -----------------------------
#     # Helper: safe log scale
#     # -----------------------------
#     def safe(values):
#         return [max(v, 1e-8) for v in values]

#     # =============================
#     # TOTAL LOSS
#     # =============================
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, safe(history["train_total"]), label="Train")
#     plt.plot(epochs, safe(history["val_total"]), label="Val")
#     plt.yscale("log")
#     plt.xlabel("Epoch")
#     plt.ylabel("Total Loss (log)")
#     plt.title("Total Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(LOG_DIR / "loss_total.png")
#     plt.close()

#     # =============================
#     # SPAN MLM LOSS
#     # =============================
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, safe(history["train_span_mlm"]), label="Train")
#     plt.plot(epochs, safe(history["val_span_mlm"]), label="Val")
#     plt.yscale("log")
#     plt.xlabel("Epoch")
#     plt.ylabel("SpanMLM Loss (log)")
#     plt.title("SpanMLM Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(LOG_DIR / "loss_span_mlm.png")
#     plt.close()

#     # =============================
#     # NT-XENT LOSS
#     # =============================
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, safe(history["train_nt_xent"]), label="Train")
#     plt.plot(epochs, safe(history["val_nt_xent"]), label="Val")
#     plt.yscale("log")
#     plt.xlabel("Epoch")
#     plt.ylabel("NT-Xent Loss (log)")
#     plt.title("Contrastive NT-Xent Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(LOG_DIR / "loss_nt_xent.png")
#     plt.close()

#     # =============================
#     # UNCERTAINTY PARAMETERS
#     # =============================
#     if "log_var_span" in history:
#         plt.figure(figsize=(8, 5))
#         plt.plot(epochs, history["log_var_span"], label="log σ² (SpanMLM)")
#         plt.plot(epochs, history["log_var_nt"], label="log σ² (NT-Xent)")
#         plt.xlabel("Epoch")
#         plt.ylabel("Log Variance")
#         plt.title("Uncertainty Weights")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(LOG_DIR / "uncertainty.png")
#         plt.close()

#     print("✅ Loss plots saved to:", LOG_DIR)


# if __name__ == "__main__":
#     plot_losses()
import json
from pathlib import Path
import matplotlib.pyplot as plt
from m_garud.config import LOG_DIR

def plot_losses():
    history_path = LOG_DIR / "loss_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} not found")

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_total"]) + 1)

    # Helper: safe log scale to avoid math errors with zero/negative values
    def safe(values):
        return [max(v, 1e-8) for v in values]

    # =============================
    # 1. TOTAL & MACRO LOSS
    # =============================
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, safe(history["train_total"]), 'b--', label="Train Total", alpha=0.6)
    plt.plot(epochs, safe(history["val_total"]), 'b-', label="Val Total", linewidth=2)
    
    if "macro_val_loss" in history:
        plt.plot(epochs, safe(history["macro_val_loss"]), 'r-', label="Macro-Val (Balanced)", linewidth=2.5)
        
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Combined Total & Macro Validation Loss")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(LOG_DIR / "loss_total_macro.png")
    plt.close()

    # =============================
    # 2. LANGUAGE-WISE LOSSES (DYNAMIC)
    # =============================
    # Automatically finds keys like val_loss_bhili, val_loss_gondi, etc.
    lang_keys = [k for k in history.keys() if k.startswith("val_loss_")]
    
    if lang_keys:
        plt.figure(figsize=(10, 6))
        for key in lang_keys:
            lang_name = key.replace("val_loss_", "").capitalize()
            plt.plot(epochs, safe(history[key]), label=f"Val Loss: {lang_name}")
            
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log)")
        plt.title("Language-Specific Validation Performance")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(LOG_DIR / "loss_by_language.png")
        plt.close()

    # =============================
    # 3. UNCERTAINTY WEIGHTS (σ²)
    # =============================
    # Handles both 'log_var_span' and 'span_uw' key variations
    span_key = "span_uw" if "span_uw" in history else "log_var_span"
    nt_key = "nt_uw" if "nt_uw" in history else "log_var_nt"

    if span_key in history and nt_key in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history[span_key], 'g-', label="log σ² (SpanMLM)")
        plt.plot(epochs, history[nt_key], 'm-', label="log σ² (NT-Xent)")
        plt.xlabel("Epoch")
        plt.ylabel("Log Variance (Weighting Parameter)")
        plt.title("Learned Multi-Task Uncertainty Weights")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(LOG_DIR / "uncertainty_weights.png")
        plt.close()

    # =============================
    # 4. TASK-SPECIFIC LOSSES
    # =============================
    for task in ["span_mlm", "nt_xent"]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, safe(history[f"train_{task}"]), label="Train")
        plt.plot(epochs, safe(history[f"val_{task}"]), label="Val")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel(f"{task.upper()} Loss (log)")
        plt.title(f"{task.replace('_', ' ').upper()} Loss Trend")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(LOG_DIR / f"loss_{task}.png")
        plt.close()

    print(f"✅ Enhanced loss plots successfully saved to: {LOG_DIR}")

if __name__ == "__main__":
    plot_losses()