import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from lightly.loss import NTXentLoss


# ------------------------------------------------------------------
#  MoCo + De-biased NT-Xent  (small-batch friendly)
# ------------------------------------------------------------------
# class DebiasedNTXent(nn.Module):
#     """
#     MoCo-style memory bank + de-biased NT-Xent for small batches.
#     Works with [CLS] embeddings (no extra gradient graph kept for queue).
#     """

#     def __init__(self,
#                  feature_dim: int = 768,
#                  queue_size: int = 8192,
#                  temperature: float = 0.1,
#                  device: str = "cpu"):
#         super().__init__()
#         self.temperature = temperature
#         self.queue_size = queue_size
#         self.device = device

#         # queue: (K, D)  – Hindi embeddings
#         self.register_buffer("queue", F.normalize(torch.randn(queue_size, feature_dim), dim=1))
#         self.queue = self.queue.to(self.device)        # <- NEW

#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys: torch.Tensor):
#         """keys: (B, D); already detached & L2-normalised"""
#         keys = keys.detach()
#         batch_size = keys.size(0)

#         ptr = int(self.queue_ptr)
#         # replace the oldest batch in the circular queue
#         if ptr + batch_size <= self.queue_size:
#             self.queue[ptr:ptr + batch_size] = keys
#         else:
#             wrap = ptr + batch_size - self.queue_size
#             self.queue[ptr:] = keys[:self.queue_size - ptr]
#             self.queue[:wrap] = keys[self.queue_size - ptr:]
#         ptr = (ptr + batch_size) % self.queue_size
#         self.queue_ptr[0] = ptr

#     def forward(self, z_student: torch.Tensor, z_teacher: torch.Tensor):
#         """
#         Inputs:
#             z_student: (B, D)  – Bhili [CLS] embeddings
#             z_teacher: (B, D)  – Hindi [CLS] embeddings
#         Returns scalar loss.
#         """
#         # --- normalise once ---
#         # z_s = F.normalize(z_student, dim=1)
#         # z_t = F.normalize(z_teacher, dim=1)
#         # z_s = z_student
#         # z_t = z_teacher

#         # batch_size, dim = z_s.shape
#         # K = self.queue_size

#         # # positives: diagonal of batch x batch
#         # l_pos = torch.einsum("bd,bd->b", z_s, z_t).unsqueeze(-1)  # (B,1)

#         # # negatives: batch x queue
#         # queue = self.queue.clone().detach()           # NEW
#         # l_neg = torch.einsum("bd,kd->bk", z_s, queue) # uses detached buffer, (B,K)

#         # # de-biased denominator term (Chuang et al. 2020)
#         # # assume τ_pos = τ_neg = self.temperature
#         # tau = self.temperature
#         # logits = torch.cat([l_pos, l_neg], dim=1) / tau           # (B, 1+K)
#         # labels = torch.zeros(batch_size, dtype=torch.long, device=z_s.device)

#         # # de-biased correction: subtract expected false-negative mass
#         # # simplified estimator for small α (probability a random negative is actually positive)
#         # # here α = 1/K  (uniform prior)
#         # alpha = 1.0 / K
#         # debiased_neg = l_neg - alpha * torch.exp(l_pos / tau)
#         # logits_debiased = torch.cat([l_pos, debiased_neg], dim=1) / tau

#         # loss = F.cross_entropy(logits_debiased, labels)

#         # # update memory bank with *teacher* embeddings
#         # self._dequeue_and_enqueue(z_t.detach())

#         # return loss

#         z_s = z_student
#         z_t = z_teacher.detach()          # no grads into teacher
#         l_pos = torch.einsum("bd,bd->b", z_s, z_t).unsqueeze(-1)          # (B, 1)
#         l_neg = torch.einsum("bd,kd->bk", z_s, self.queue.clone().detach())  # (B, K)
#         logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature      # (B, 1+K)
#         labels = torch.zeros(z_s.size(0), dtype=torch.long, device=z_s.device)
#         loss = F.cross_entropy(logits, labels)
#         self._dequeue_and_enqueue(z_t.detach())
#         return loss


class DebiasedNTXent(nn.Module):
    """
    MoCo-style queue + De-biased NT-Xent (Chuang et al. 2020).
    Works with [CLS] embeddings.

    Args
    ----
    feature_dim : int
        Dimensionality of the embeddings (default 768 for MuRIL).
    queue_size  : int
        Size of the memory bank (K).
    temperature : float
        Temperature τ for NT-Xent.
    alpha       : float
        Prior probability that a random negative is actually positive.
        Usually set to 1 / queue_size for a uniform prior.
    device      : str or torch.device
        Device on which the queue lives.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        queue_size: int = 8192,
        temperature: float = 0.1,
        alpha: float = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.alpha = alpha if alpha is not None else 1.0 / queue_size

        # Memory bank (K, D) – always detached
        queue = torch.randn(queue_size, feature_dim, device=device)
        queue = F.normalize(queue, dim=-1)
        self.register_buffer("queue", queue)

        # Circular pointer
        ptr = torch.zeros(1, dtype=torch.long, device=device)
        self.register_buffer("ptr", ptr)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor):
        """Append a batch of keys (B, D) to the FIFO queue."""
        keys = keys.detach()
        b = keys.size(0)
        ptr = int(self.ptr)

        # Wrap-around write
        if ptr + b <= self.queue_size:
            self.queue[ptr : ptr + b] = keys
        else:
            rest = self.queue_size - ptr
            self.queue[ptr:] = keys[:rest]
            self.queue[: b - rest] = keys[rest:]
        self.ptr[0] = (ptr + b) % self.queue_size

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        z_s : (B, D)  – student (Bhili) embeddings
        z_t : (B, D)  – teacher (Hindi) embeddings

        Returns scalar loss.
        """
        B, D = z_s.shape
        device = z_s.device

        z_s = F.normalize(z_s, dim=-1)
        z_t = F.normalize(z_t, dim=-1).detach()
        # queue = F.normalize(self.queue.clone(), dim=-1).to(device)
        # Ensure embeddings are on the same device as the queue
        # queue = self.queue.clone().to(device)  # (K, D)

        # Dot products
        l_pos = torch.einsum("bd,bd->b", z_s, z_t)          # (B,)
        # l_neg = torch.einsum("bd,kd->bk", z_s, queue)       # (B, K)
        l_neg = torch.einsum("bd,kd->bk", z_s, self.queue.clone().detach())  # (B, K)

        # De-biased logits
        tau = self.temperature
        logits_pos = l_pos.unsqueeze(-1) / tau              # (B, 1)
        logits_neg = l_neg / tau                            # (B, K)

        # Correction term: subtract expected false-negative mass
        # α * exp(l_pos / τ)
        # correction = self.alpha * torch.exp(logits_pos)     # (B, 1)
        # logits_neg = logits_neg - correction                # (B, K)

        logits = torch.cat([logits_pos, logits_neg], dim=1) # (B, 1+K)
        labels = torch.zeros(B, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._enqueue(z_t)

        return loss


class ParallelDataset(Dataset):
    """
    Expects lists of Hindi and Bhili sentences.
    """
    def __init__(self, hindi_texts, bhili_texts):
        self.hi = hindi_texts
        self.bh = bhili_texts

    def __len__(self):  
        return len(self.hi)

    def __getitem__(self, idx):
        return {"hi": self.hi[idx], "bh": self.bh[idx]}


class DistillCollatorCustom:
    """
    Encodes Hindi with teacher tokenizer (no masking).
    Encodes Bhili with student tokenizer and applies MLM masking via DataCollator.
    """
    def __init__(self, teacher_tok, student_tok, max_len=128, mlm_prob=0.15, device="cpu"):
        self.teacher_tok = teacher_tok
        self.student_tok = student_tok
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.device = device
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tok,
            mlm=True,
            mlm_probability=mlm_prob,
            return_tensors="pt"
        )

    def __call__(self, batch):
        # Hindi (teacher)
        hi_sentences = [item["hi"] for item in batch]
        hi_enc = self.teacher_tok(
            hi_sentences,
            padding='longest',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Bhili (student) with MLM masking
        bh_sentences = [item["bh"] for item in batch]
        bh_enc = self.student_tok(
            bh_sentences,
            padding='longest',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Apply MLM masking
        bh_inputs = [
            {
                "input_ids": bh_enc["input_ids"][i],  # Shape: [seq_len]
                "attention_mask": bh_enc["attention_mask"][i]  # Shape: [seq_len]
            }
            for i in range(bh_enc["input_ids"].shape[0])
        ]
        # Apply MLM masking
        bh_mlm = self.mlm_collator(bh_inputs)
        
        return {
            "hi_input_ids": hi_enc["input_ids"],
            "hi_attention_mask": hi_enc["attention_mask"],
            "bh_input_ids": bh_mlm["input_ids"],
            "bh_attention_mask": bh_mlm["attention_mask"],
            "bh_labels": bh_mlm["labels"]
        }