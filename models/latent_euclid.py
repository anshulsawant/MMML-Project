import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoConfig

class LatentPredictor(nn.Module):
    """
    2-Layer MLP translating the Qwen3-VL X-Encoder hidden states 
    to the target Qwen3-0.6B Y-Encoder manifold dimensions.
    """
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_features, out_features)
            
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x):
        return self.mlp(x)

def setup_latent_euclid_tokenizer(model_id: str = "Qwen/Qwen3-VL-4B-Instruct", k_steps: int = 4):
    """
    Loads tokenizer and adds the new <thought_1>...<thought_k> tokens.
    Requires resizing the model embeddings afterwards.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    thought_tokens = [f"<thought_{i+1}>" for i in range(k_steps)]
    num_added = tokenizer.add_tokens(thought_tokens, special_tokens=True)
    
    print(f"Added {num_added} new thought tokens: {thought_tokens}")
    
    # Cache the token IDs for fast indexing during the forward pass
    thought_token_ids = tokenizer.convert_tokens_to_ids(thought_tokens)
    
    return tokenizer, thought_token_ids

class LatentEuclid(nn.Module):
    def __init__(self, 
                 base_model_id: str = "Qwen/Qwen3-VL-4B-Instruct", 
                 target_model_id: str = "Qwen/Qwen3-0.6B",
                 k_steps: int = 4):
        super().__init__()
        
        # Dynamically fetch the target model's hidden dimension
        target_config = AutoConfig.from_pretrained(target_model_id)
        target_dim = getattr(target_config, "hidden_size", getattr(getattr(target_config, "text_config", None), "hidden_size", 1024))
        
        # 1. Setup Tokenizer & Model
        self.tokenizer, self.thought_ids = setup_latent_euclid_tokenizer(base_model_id, k_steps)
        self.k_steps = k_steps
        
        print(f"Loading Base LatentEuclid Vision-Language Model ({base_model_id})...")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 2. Resize Embeddings to accommodate the new <thought> tokens
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        base_hidden_size = getattr(self.vlm.config, "hidden_size", getattr(getattr(self.vlm.config, "text_config", None), "hidden_size", None))
        
        # 3. Predictor Head
        # Maps the VLM's massive hidden state down to the small target LLM's dimensionality
        self.predictor = LatentPredictor(
            in_features=base_hidden_size, 
            out_features=target_dim
        ).to(self.vlm.device).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None):
        """
        Parallel Forward Pass executing the VL-JEPA extraction.
        Since Qwen is a Causal Language Model, the lower-triangular mask natively
        ensures <thought_2> sees <thought_1>, but not <thought_3>.
        """
        # 1. Run the base VLM forward pass
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_states = outputs.hidden_states[-1] # [batch, seq_len, hidden_size]
        
        batch_size = input_ids.shape[0]
        predicted_targets = []
        
        # 2. Extract hidden states strictly at the positions of the <thought> tokens
        for b in range(batch_size):
            b_input_ids = input_ids[b]
            b_hidden = last_hidden_states[b]
            
            # Find the indices of our K thought tokens in this sequence
            # (Assuming strict ordering: thought_1, thought_2, etc. at the end of the sequence)
            thought_positions = []
            for t_id in self.thought_ids:
                pos = (b_input_ids == t_id).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    raise ValueError(f"Thought token {t_id} missing from input sequence.")
                thought_positions.append(pos[-1]) # take the last occurrence if duplicated
            
            # Extract the raw VLM vectors: shape [K, hidden_size]
            thought_vectors = b_hidden[torch.stack(thought_positions)]
            predicted_targets.append(thought_vectors)
            
        # Shape: [batch, K, hidden_size]
        predicted_targets = torch.stack(predicted_targets)
        
        # 3. Project to Target Manifold
        # Shape: [batch, K, target_dim]
        projected_latents = self.predictor(predicted_targets)
        
        return projected_latents

if __name__ == "__main__":
    # Test Scaffold
    # model = LatentEuclid()
    pass
