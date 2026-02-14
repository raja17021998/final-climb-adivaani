import torch
import json
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Import your actual student architecture to ensure state_dict matches
# (Assuming the structure we discussed previously)
from m_garud.models.student.student_encoder import StudentEncoder
from m_garud.config import DEVICE, LOG_DIR

class EmbeddingExtractor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        
        # 1. Load Config
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)
            
        # 2. Initialize Tokenizer & Backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["mbert_path"])
        
        # 3. Build Student Architecture
        # Ensure parameters match what was used during training
        self.model = StudentEncoder(
            mbert_path=self.config["mbert_path"],
            num_layers=self.config["num_layers"],
            hidden_dim=self.config["hidden_dim"],
            num_heads=self.config["num_heads"],
            num_kv_groups=self.config["num_kv_groups"]
        ).to(DEVICE)
        
        # 4. Load Saved Trainable Weights (Safetensors)
        weights = load_file(self.model_dir / "model.safetensors")
        
        # Strict=False because safetensors only contains trainable layers, 
        # while mBERT backbone is loaded via from_pretrained.
        self.model.load_state_dict(weights, strict=False)
        self.model.eval()
        print(f"✅ Model loaded from {self.model_dir}")

    @torch.no_grad()
    def get_embeddings(self, sentences, tribal_lang, batch_size=32):
        """
        Generates mean-pooled embeddings for a list of tribal sentences.
        """
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(DEVICE)
            
            # Forward pass through the student (includes the custom transformer layers)
            hidden_states = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                tribal_lang=tribal_lang
            )
            
            # Mean Pooling: Only averaging non-pad tokens
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_pooled.cpu())
            
        return torch.cat(all_embeddings, dim=0)

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    extractor = EmbeddingExtractor(LOG_DIR / "final_student_model")
    
    bhili_sentences = [
        "तुमरो नाम काई छे?", # What is your name?
        "मुँह घर जाह रयूँ छूँ।" # I am going home.
    ]
    
    # Extract
    embeddings = extractor.get_embeddings(bhili_sentences, tribal_lang="bhili")
    
    print(f"Generated {embeddings.shape[0]} embeddings of size {embeddings.shape[1]}")
    
    # Save for downstream evaluation (e.g., retrieval or visualization)
    torch.save(embeddings, LOG_DIR / "bhili_test_embeddings.pt")