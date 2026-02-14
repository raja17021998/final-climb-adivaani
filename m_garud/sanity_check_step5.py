import torch

# =========================
# IMPORTS
# =========================

from m_garud.data.alternating_loader import AlternatingTribalLoader

from m_garud.models.student.student_encoder import StudentEncoder
from m_garud.models.pivot.pivot_attention import PivotAttention
from m_garud.models.pivot.teacher_encoder import FrozenTeacherEncoder

from m_garud.models.fusion.convex_fusion import ConvexTeacherFusion
from m_garud.models.losses.uncertainty_weighted_loss import (
    UncertaintyWeightedLoss,
)
import torch.nn.functional as F

from m_garud.training.moco_queue import MoCoQueue
from m_garud.config import (
    # Paths & language metadata
    LANGUAGE_MAP,
    BERT_BASE_DIR,
    TEACHER_BERT_PATHS,

    # Architecture
    HIDDEN_DIM,
    NUM_HEADS,
    NUM_KV_GROUPS,

    # Tokenization
    PAD_TOKEN_ID,

    # Training / sanity
    BATCH_SIZE,
    FUSION_TEMPERATURE,

    CONTRASTIVE_TEMPERATURE,
    MOCO_QUEUE_SIZE,
    MOCO_EMB_DIM,
    MOCO_PRIMARY_TEACHER,

    # DEVICE
    DEVICE,
)

# =========================
# DEVICE
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nRunning sanity check on device: {DEVICE}")

# ============================================================
# 1. DATA LOADER CHECK
# ============================================================

print("\n[1] DATA LOADER CHECK")

loader = AlternatingTribalLoader(
    tribal_languages=list(LANGUAGE_MAP.keys()),
    batch_size=BATCH_SIZE,
)

batch = next(iter(loader))

print("Tribal language:", batch["tribal_lang"])
print("Student input shape:", batch["student_input_ids"].shape)

assert "student_input_ids" in batch
assert "teacher_input_ids" in batch
assert "tribal_lang" in batch

# ============================================================
# 2. STUDENT ENCODER CHECK
# ============================================================

print("\n[2] STUDENT ENCODER CHECK")

student_encoder = StudentEncoder(
    mbert_path=BERT_BASE_DIR / "mbert",
    hidden_dim=HIDDEN_DIM,
).to(DEVICE)

student_encoder.eval()

student_input_ids = batch["student_input_ids"].to(DEVICE)
student_attention_mask = (student_input_ids != PAD_TOKEN_ID).long()

with torch.no_grad():
    student_h = student_encoder(
        input_ids=student_input_ids,
        attention_mask=student_attention_mask,
        tribal_lang=batch["tribal_lang"],
    )

print("Student hidden shape:", student_h.shape)

assert student_h.shape[-1] == HIDDEN_DIM
assert not torch.isnan(student_h).any()

# ============================================================
# 3. TEACHER ENCODER CHECK
# ============================================================

print("\n[3] TEACHER ENCODER CHECK")

teacher_outputs = {}

for teacher_lang, teacher_ids in batch["teacher_input_ids"].items():

    if teacher_lang not in TEACHER_BERT_PATHS:
        raise KeyError(
            f"Teacher path not defined for language: {teacher_lang}"
        )

    teacher_encoder = FrozenTeacherEncoder(
        TEACHER_BERT_PATHS[teacher_lang]
    ).to(DEVICE)

    teacher_encoder.eval()

    teacher_ids = teacher_ids.to(DEVICE)
    teacher_mask = (teacher_ids != PAD_TOKEN_ID).long()

    with torch.no_grad():
        teacher_h = teacher_encoder(
            teacher_ids,
            teacher_mask,
        )

    teacher_outputs[teacher_lang] = {
        "hidden": teacher_h,
        "mask": teacher_mask,
    }

    print(
        f"Teacher {teacher_lang} hidden shape:",
        teacher_h.shape
    )

    assert not torch.isnan(teacher_h).any()

# ============================================================
# 4. PIVOT ATTENTION CHECK (MASK-AWARE)
# ============================================================

print("\n[4] PIVOT ATTENTION CHECK (MASK-AWARE)")

pivot = PivotAttention(
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_kv_groups=NUM_KV_GROUPS,
).to(DEVICE)

pivot.eval()

pivot_outputs = {}

for teacher_lang, data in teacher_outputs.items():
    teacher_h = data["hidden"]
    teacher_mask = data["mask"]

    with torch.no_grad():
        pivot_out = pivot(
            student_h,
            teacher_h,
            teacher_mask=teacher_mask,
        )

    print(
        f"Pivot output ({teacher_lang}) shape:",
        pivot_out.shape
    )

    assert pivot_out.shape == student_h.shape
    assert not torch.isnan(pivot_out).any()

    pivot_outputs[teacher_lang] = pivot_out

# ============================================================
# 5. GRADIENT FLOW CHECK (STUDENT + PIVOT)
# ============================================================

print("\n[5] GRADIENT FLOW CHECK")

student_encoder.train()
pivot.train()

# ---- recompute with gradients ----
student_h = student_encoder(
    input_ids=student_input_ids,
    attention_mask=student_attention_mask,
    tribal_lang=batch["tribal_lang"],
)

teacher_lang = next(iter(teacher_outputs))
teacher_h = teacher_outputs[teacher_lang]["hidden"]
teacher_mask = teacher_outputs[teacher_lang]["mask"]

pivot_out = pivot(
    student_h,
    teacher_h,
    teacher_mask=teacher_mask,
)

loss = pivot_out.mean()
loss.backward()

# ---- mBERT must be frozen ----
assert not any(
    p.requires_grad for p in student_encoder.mbert.parameters()
), "mBERT parameters are NOT frozen!"

# ---- Pivot must receive gradients ----
assert any(
    p.grad is not None for p in pivot.parameters()
), "Pivot Attention received NO gradients!"

print("✔ Gradient flow verified")

# ============================================================
# 6. CONVEX FUSION + LOSS SANITY CHECK
# ============================================================

print("\n[6] CONVEX FUSION + LOSS SANITY CHECK")

student_encoder.train()
pivot.train()

student_h = student_encoder(
    input_ids=student_input_ids,
    attention_mask=student_attention_mask,
    tribal_lang=batch["tribal_lang"],
)

teacher_h_list = []

for teacher_lang, data in teacher_outputs.items():
    pivot_out = pivot(
        student_h,
        data["hidden"],
        teacher_mask=data["mask"],
    )
    teacher_h_list.append(pivot_out)

fusion = ConvexTeacherFusion(
    hidden_dim=HIDDEN_DIM,
    num_teachers=len(teacher_h_list),
    temperature=FUSION_TEMPERATURE,
).to(DEVICE)

fused_h, fusion_weights = fusion(
    student_h,
    teacher_h_list,
)

print("Fused hidden shape:", fused_h.shape)
print("Fusion weights shape:", fusion_weights.shape)

# ---- checks ----
assert fused_h.shape == student_h.shape
assert fusion_weights.shape[-1] == len(teacher_h_list) + 1

weight_sum = fusion_weights.sum(dim=-1)
assert torch.allclose(
    weight_sum,
    torch.ones_like(weight_sum),
    atol=1e-5,
), "Fusion weights are not convex!"

assert not torch.isnan(fused_h).any()

# ---- dummy losses ----
mlm_loss = fused_h.mean()
nt_loss = student_h.mean()

uw_loss = UncertaintyWeightedLoss().to(DEVICE)
total_loss = uw_loss(mlm_loss, nt_loss)

total_loss.backward()

# ---- gradient checks ----
assert any(
    p.grad is not None for p in fusion.parameters()
), "No gradients in Convex Fusion!"

assert any(
    p.grad is not None for p in student_encoder.parameters()
), "No gradients in Student Encoder!"

print("\n✅ SANITY CHECK PASSED (STEP-5 + STEP-6 ARE TRAINING-SAFE)")


# ============================================================
# 7. STEP-7 TRAINING LOGIC SANITY CHECK (MoCo + UW Loss)
# ============================================================

print("\n[7] STEP-7 TRAINING LOGIC SANITY CHECK")

student_encoder.train()
pivot.train()
fusion.train()

# ------------------------------------------------------------
# MoCo Queue (persistent negatives)
# ------------------------------------------------------------

moco_queue = MoCoQueue(
    queue_size=MOCO_QUEUE_SIZE,
    emb_dim=MOCO_EMB_DIM,
).to(DEVICE)

# ------------------------------------------------------------
# Forward pass (same as training)
# ------------------------------------------------------------

student_h = student_encoder(
    input_ids=student_input_ids,
    attention_mask=student_attention_mask,
    tribal_lang=batch["tribal_lang"],
)

# ------------------------------------------------------------
# Teacher → Pivot
# ------------------------------------------------------------

teacher_pivot_outputs = {}

for teacher_lang, data in teacher_outputs.items():
    teacher_pivot_outputs[teacher_lang] = pivot(
        student_h,
        data["hidden"],
        teacher_mask=data["mask"],
    )

# ------------------------------------------------------------
# Convex fusion
# ------------------------------------------------------------

fused_h, fusion_weights = fusion(
    student_h,
    list(teacher_pivot_outputs.values()),
)

# ------------------------------------------------------------
# SpanMLM placeholder (structural)
# ------------------------------------------------------------

span_mlm_loss = fused_h.mean()

# ------------------------------------------------------------
# MoCo-style NT-Xent (Hindi anchor)
# ------------------------------------------------------------

tribal_sent = F.normalize(
    fused_h.mean(dim=1), dim=-1
)  # [B, D]

with torch.no_grad():
    hindi_pivot = teacher_pivot_outputs[MOCO_PRIMARY_TEACHER]
    teacher_sent = F.normalize(
        hindi_pivot.mean(dim=1), dim=-1
    )

# ---- positives ----
l_pos = torch.einsum(
    "nd,nd->n", tribal_sent, teacher_sent
).unsqueeze(-1)

# ---- negatives (READ-ONLY queue) ----
negatives = moco_queue.queue.detach()   # ← IMPORTANT
l_neg = torch.einsum(
    "nd,kd->nk", tribal_sent, negatives
)

logits = torch.cat([l_pos, l_neg], dim=1)
logits /= CONTRASTIVE_TEMPERATURE

labels = torch.zeros(
    logits.size(0),
    dtype=torch.long,
    device=logits.device,
)

nt_loss = F.cross_entropy(logits, labels)

# ------------------------------------------------------------
# Uncertainty-weighted total loss
# ------------------------------------------------------------

uw_loss = UncertaintyWeightedLoss().to(DEVICE)
total_loss = uw_loss(span_mlm_loss, nt_loss)

print("SpanMLM loss:", span_mlm_loss.item())
print("NT-Xent loss:", nt_loss.item())
print("Total loss:", total_loss.item())

# ------------------------------------------------------------
# Backprop (MUST happen BEFORE queue update)
# ------------------------------------------------------------

total_loss.backward()

# ------------------------------------------------------------
# NOW update MoCo queue (SAFE)
# ------------------------------------------------------------

moco_queue.enqueue(teacher_sent)

# ------------------------------------------------------------
# Gradient checks (THE POINT OF STEP-7)
# ------------------------------------------------------------

# Student must receive gradients
assert any(
    p.grad is not None for p in student_encoder.parameters()
), "❌ No gradients in Student Encoder!"

# Pivot must receive gradients
assert any(
    p.grad is not None for p in pivot.parameters()
), "❌ No gradients in Pivot Attention!"

# Fusion must receive gradients
assert any(
    p.grad is not None for p in fusion.parameters()
), "❌ No gradients in Convex Fusion!"

# Teacher embeddings must be frozen
assert not teacher_sent.requires_grad, \
    "❌ Teacher embeddings should be frozen!"

# MoCo queue must be gradient-isolated
assert moco_queue.queue.grad is None, \
    "❌ MoCo queue received gradients!"

print("\n✅ STEP-7 SANITY CHECK PASSED (FULL TRAINING LOGIC VERIFIED)")

# ============================================================
# 8A. SPAN MASKING + SPAN MLM SANITY CHECK
# ============================================================

print("\n[8A] SPAN MASKING + SPAN MLM SANITY CHECK")

from m_garud.training.span_masking import generate_span_mask
from m_garud.training.span_mlm_loss import span_mlm_loss

student_encoder.train()

# ------------------------------------------------------------
# Generate span mask
# ------------------------------------------------------------
mask_positions = generate_span_mask(student_input_ids)

print("Mask positions shape:", mask_positions.shape)

# mask must be boolean
assert mask_positions.dtype == torch.bool, \
    "Span mask must be boolean!"

# mask must not be empty
assert mask_positions.any(), \
    "❌ Span mask is empty — masking logic broken!"

# mask must align with input
assert mask_positions.shape == student_input_ids.shape, \
    "❌ Mask shape does not match input shape!"

print(
    "Masked tokens:",
    mask_positions.sum().item(),
    "/",
    mask_positions.numel(),
)

# ------------------------------------------------------------
# Student forward WITH logits
# ------------------------------------------------------------
student_h, student_logits = student_encoder(
    input_ids=student_input_ids,
    attention_mask=student_attention_mask,
    tribal_lang=batch["tribal_lang"],
    return_logits=True,
)

print("Student logits shape:", student_logits.shape)

# ------------------------------------------------------------
# SpanMLM loss
# ------------------------------------------------------------
mlm_loss = span_mlm_loss(
    logits=student_logits,
    target_ids=student_input_ids,
    mask_positions=mask_positions,
)

print("SpanMLM loss:", mlm_loss.item())

# loss must be scalar & finite
assert mlm_loss.dim() == 0, \
    "❌ MLM loss must be scalar!"

assert torch.isfinite(mlm_loss), \
    "❌ MLM loss is NaN or Inf!"

# ------------------------------------------------------------
# Backprop check
# ------------------------------------------------------------
student_encoder.zero_grad()
mlm_loss.backward()

# student must receive gradients
assert any(
    p.grad is not None for p in student_encoder.parameters()
), "❌ No gradients in Student Encoder from SpanMLM!"

print("\n✅ STEP-8A SANITY CHECK PASSED (SPAN MASKING + MLM VERIFIED)")


# ============================================================
# 8B. CROSS-LINGUAL RETRIEVAL SANITY CHECK
# ============================================================


from m_garud.models.pivot.teacher_encoder import FrozenTeacherEncoder
from m_garud.config import TEACHER_BERT_PATHS

# ============================================================
# Build frozen teacher encoders (ONCE)
# ============================================================

teacher_encoders = {}

for lang, path in TEACHER_BERT_PATHS.items():
    enc = FrozenTeacherEncoder(path).to(DEVICE)
    enc.eval()

    for p in enc.parameters():
        p.requires_grad = False

    teacher_encoders[lang] = enc


from m_garud.evaluation.cross_lingual_retrieval import (
    CrossLingualRetrievalEvaluator
)

print("\n[8B] CROSS-LINGUAL RETRIEVAL SANITY CHECK (ALL LANGUAGES)")

evaluator = CrossLingualRetrievalEvaluator(
    student=student_encoder,
    pivot=pivot,
    fusion=fusion,
    teacher_encoders=teacher_encoders,
    device=DEVICE,
)

all_metrics = evaluator.evaluate_all(loader)

for tribal, metrics in all_metrics.items():
    print(f"\nTribal language: {tribal}")

    print("Similarity scores:")
    for k, v in metrics["similarity_scores"].items():
        print(f"  {k}: {v:.4f}")

    print("\nMean fusion weights:")
    for k, v in metrics["fusion_weights"].items():
        print(f"  {k}: {v:.4f}")

print("\n✅ STEP-8B SANITY CHECK PASSED (ALL TRIBAL LANGUAGES)")
