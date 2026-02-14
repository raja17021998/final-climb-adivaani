


# import torch
# from collections import defaultdict
# from tqdm import tqdm
# import torch.nn.functional as F

# from m_garud.config import (
#     DEVICE,
#     PAD_TOKEN_ID,
#     BATCH_SIZE,
#     CONTRASTIVE_TEMPERATURE,
#     MOCO_PRIMARY_TEACHER,
# )

# from m_garud.training.span_masking import generate_span_mask
# from m_garud.training.span_mlm_loss import span_mlm_loss

# @torch.no_grad()
# def evaluate_epoch(trainer, dataloader):
#     """
#     Validation loop that dynamically tracks losses per tribal language.
#     """
#     trainer.student.eval()
#     trainer.pivot.eval()
#     trainer.fusion.eval()

#     # General metrics across all batches
#     epoch_losses = defaultdict(list)
#     # Specific tracking per language for Macro-Average Early Stopping
#     lang_total_losses = defaultdict(list)

#     # Calculate steps based on the validation data available
#     steps = min(
#         len(ds) // BATCH_SIZE
#         for ds in dataloader.datasets.values()
#     ) * len(dataloader.datasets)

#     for _ in tqdm(range(steps), desc="Validation"):
#         batch = next(dataloader)
#         current_lang = batch["tribal_lang"]

#         # ==================================================
#         # Student forward
#         # ==================================================
#         student_ids = batch["student_input_ids"].to(DEVICE)
#         student_mask = (student_ids != PAD_TOKEN_ID).long()

#         student_h = trainer.student(
#             input_ids=student_ids,
#             attention_mask=student_mask,
#             tribal_lang=current_lang,
#         )

#         # ==================================================
#         # Teacher → Pivot
#         # ==================================================
#         pivot_outputs = {}
#         for t_lang, teacher_ids in batch["teacher_input_ids"].items():
#             teacher_ids = teacher_ids.to(DEVICE)
#             teacher_mask = (teacher_ids != PAD_TOKEN_ID).long()

#             teacher_h = trainer.teacher_encoders[t_lang](
#                 teacher_ids,
#                 teacher_mask,
#             )

#             pivot_outputs[t_lang] = trainer.pivot(
#                 student_h,
#                 teacher_h,
#                 teacher_mask=teacher_mask,
#             )

#         # ==================================================
#         # Convex fusion
#         # ==================================================
#         fused_h, _ = trainer.fusion(
#             student_h,
#             list(pivot_outputs.values()),
#         )

#         # ==================================================
#         # SpanMLM loss
#         # ==================================================
#         mask_positions = generate_span_mask(student_ids)

#         _, student_logits = trainer.student(
#             input_ids=student_ids,
#             attention_mask=student_mask,
#             tribal_lang=current_lang,
#             return_logits=True,
#         )

#         span_loss = span_mlm_loss(
#             logits=student_logits,
#             target_ids=student_ids,
#             mask_positions=mask_positions,
#         )

#         # ==================================================
#         # NT-Xent loss (Contrastive Alignment)
#         # ==================================================
#         tribal_sent = F.normalize(fused_h.mean(dim=1), dim=-1)
#         teacher_anchor = F.normalize(pivot_outputs[MOCO_PRIMARY_TEACHER].mean(dim=1), dim=-1)

#         l_pos = torch.einsum("nd,nd->n", tribal_sent, teacher_anchor).unsqueeze(-1)
#         l_neg = torch.einsum("nd,kd->nk", tribal_sent, trainer.moco_queue.get_queue())

#         logits = torch.cat([l_pos, l_neg], dim=1)
#         logits /= CONTRASTIVE_TEMPERATURE
#         labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)

#         nt_loss = F.cross_entropy(logits, labels)

#         # ==================================================
#         # Total loss (Uncertainty weighted)
#         # ==================================================
#         total_loss = trainer.uw_loss(span_loss, nt_loss)

#         # Record metrics
#         epoch_losses["span_mlm"].append(span_loss.item())
#         epoch_losses["nt_xent"].append(nt_loss.item())
#         epoch_losses["total"].append(total_loss.item())
        
#         # Tag the total loss to the specific tribal language
#         lang_total_losses[current_lang].append(total_loss.item())

#     # Build response dictionary
#     results = {
#         f"val_{k}": sum(v) / len(v) for k, v in epoch_losses.items()
#     }
    
#     # Add per-language metrics: "val_loss_bhili", "val_loss_gondi", etc.
#     for lang, losses in lang_total_losses.items():
#         results[f"val_loss_{lang}"] = sum(losses) / len(losses)

#     return results




import torch
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from m_garud.config import DEVICE, PAD_TOKEN_ID, BATCH_SIZE, CONTRASTIVE_TEMPERATURE, MOCO_PRIMARY_TEACHER
from m_garud.training.span_masking import generate_span_mask
from m_garud.training.span_mlm_loss import span_mlm_loss

@torch.no_grad()
def evaluate_epoch(trainer, dataloader):
    trainer.student.eval()
    trainer.pivot.eval()
    trainer.fusion.eval()

    epoch_losses = defaultdict(list)
    lang_task_losses = defaultdict(list)

    steps = min(len(ds) // BATCH_SIZE for ds in dataloader.datasets.values()) * len(dataloader.datasets)

    for _ in tqdm(range(steps), desc="Validation"):
        batch = next(dataloader)
        current_lang = batch["tribal_lang"]

        # 1. Forward Pass
        student_ids = batch["student_input_ids"].to(DEVICE)
        student_mask = (student_ids != PAD_TOKEN_ID).long()
        student_h = trainer.student(input_ids=student_ids, attention_mask=student_mask, tribal_lang=current_lang)

        # 2. Alignment Path
        pivot_outputs = {}
        for t_lang, t_ids in batch["teacher_input_ids"].items():
            t_ids = t_ids.to(DEVICE)
            t_mask = (t_ids != PAD_TOKEN_ID).long()
            t_h = trainer.teacher_encoders[t_lang](t_ids, t_mask)
            pivot_outputs[t_lang] = trainer.pivot(student_h, t_h, teacher_mask=t_mask)

        fused_h, _ = trainer.fusion(student_h, list(pivot_outputs.values()))

        # 3. Loss Calculation
        mask_pos = generate_span_mask(student_ids)
        _, logits = trainer.student(input_ids=student_ids, attention_mask=student_mask, 
                                    tribal_lang=current_lang, return_logits=True)
        span_l = span_mlm_loss(logits=logits, target_ids=student_ids, mask_positions=mask_pos)

        # Contrastive NT-Xent
        tribal_v = F.normalize(fused_h.mean(dim=1), dim=-1)
        anchor_v = F.normalize(pivot_outputs[MOCO_PRIMARY_TEACHER].mean(dim=1), dim=-1)
        l_pos = torch.einsum("nd,nd->n", tribal_v, anchor_v).unsqueeze(-1)
        l_neg = torch.einsum("nd,kd->nk", tribal_v, trainer.moco_queue.get_queue())
        nt_logits = torch.cat([l_pos, l_neg], dim=1) / CONTRASTIVE_TEMPERATURE
        nt_l = F.cross_entropy(nt_logits, torch.zeros(nt_logits.size(0), dtype=torch.long, device=DEVICE))

        # Total Weight Loss (Weighted for training signal)
        total_l = trainer.uw_loss(span_l, nt_l)

        # Track Metrics
        epoch_losses["span_mlm"].append(span_l.item())
        epoch_losses["nt_xent"].append(nt_l.item())
        epoch_losses["total"].append(total_l.item())
        
        # Track raw (unweighted) total per language
        lang_task_losses[current_lang].append(span_l.item() + nt_l.item())

    results = {f"val_{k}": sum(v) / len(v) for k, v in epoch_losses.items()}
    for lang, losses in lang_task_losses.items():
        results[f"val_loss_{lang}"] = sum(losses) / len(losses)

    return results