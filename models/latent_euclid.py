'''
LatentEuclid: The X-Encoder Architecture

This module formally implements the Macro-JEPA paradigm for the LatentEuclid architecture.
Instead of relying on standard autoregressive discrete text generation, this network isolates the 
multimodal geometry representations of an input problem, projecting them into `K` continuous
"thought vectors" representing a spatial solution using a singular parallel forward pass.

Key Architectural Properties:
1. `<thought_k>` Vocabulary Injection: K new special tokens are injected to query the multimodal topology.
2. Causal Masking: Exploits the native Causal Mask to ensure temporal logic (Thought 2 sees 1, but not 3).
3. LatentPredictor MLP: The final layer hidden states are extracted over these tokens and mapped 
   to the dimensionality of the expert Y-Encoder (Qwen3-0.6B) target manifold.
'''

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoConfig, AutoProcessor

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

def setup_latent_euclid_tokenizer(model_id: str = "Qwen/Qwen3-VL-4B-Instruct", max_thought_tokens: int = 30):
    """
    Loads tokenizer and adds the new <thought_1>...<thought_k> dynamically allocated sequence tokens
    plus the <REASON> pooling anchor token used for contrastive text encoding.
    Requires resizing the model embeddings afterwards.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    thought_tokens = [f"<thought_{i+1}>" for i in range(max_thought_tokens)]
    num_added = tokenizer.add_tokens(thought_tokens, special_tokens=True)
    
    print(f"Added {num_added} dynamically routed new thought tokens")
    
    # Add the <REASON> anchor token used as a pooling sentinel for contrastive text encoding
    num_reason = tokenizer.add_tokens(["<REASON>"], special_tokens=True)
    print(f"Added {num_reason} <REASON> token(s)")
    reason_token_id = tokenizer.convert_tokens_to_ids("<REASON>")
    
    # Cache the token IDs for fast PyTorch tensorized matching during the forward pass
    thought_token_ids = tokenizer.convert_tokens_to_ids(thought_tokens)
    
    return tokenizer, thought_token_ids, reason_token_id

class LatentEuclid(nn.Module):
    def __init__(self, 
                 base_model_id: str = "Qwen/Qwen3-VL-4B-Instruct", 
                 target_model_id: str = "Qwen/Qwen3-0.6B",
                 max_thought_tokens: int = 30):
        super().__init__()
        
        # Dynamically fetch the target model's hidden dimension
        target_config = AutoConfig.from_pretrained(target_model_id)
        if hasattr(target_config, "hidden_size"):
            target_dim = target_config.hidden_size
        elif hasattr(target_config, "text_config"):
            if isinstance(target_config.text_config, dict):
                target_dim = target_config.text_config.get("hidden_size", 1024)
            else:
                target_dim = getattr(target_config.text_config, "hidden_size", 1024)
        else:
            target_dim = 1024
        
        # 1. Setup Tokenizer & Model
        self.tokenizer, self.thought_ids, self.reason_token_id = setup_latent_euclid_tokenizer(base_model_id, max_thought_tokens)
        
        # Load the multimodal processor to handle image/text inputs, embedding the custom tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(base_model_id)
            self.processor.tokenizer = self.tokenizer
        except Exception as e:
            print(f"Warning: Could not load AutoProcessor for {base_model_id}: {e}")
            self.processor = None
            
        
        print(f"Loading Base LatentEuclid Vision-Language Model ({base_model_id})...")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto"
        )
        
        # 2. Resize Embeddings to accommodate the new <thought> tokens
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.vlm.config, "hidden_size"):
            base_hidden_size = self.vlm.config.hidden_size
        elif hasattr(self.vlm.config, "text_config"):
            if isinstance(self.vlm.config.text_config, dict):
                base_hidden_size = self.vlm.config.text_config.get("hidden_size", 2560)
            else:
                base_hidden_size = getattr(self.vlm.config.text_config, "hidden_size", 2560)
        else:
            base_hidden_size = 2560
        
        # 3. Predictor Head
        # Maps the VLM's massive hidden state down to the small target LLM's dimensionality
        self.predictor = LatentPredictor(
            in_features=base_hidden_size, 
            out_features=target_dim
        ).to(self.vlm.device).to(torch.bfloat16)

        # 4. Standardize the Config API for Downstream Pipelines 
        # (Exposing `.config.hidden_size` correctly scaled to the Predictor Output constraints)
        class LatentEuclidConfig: pass
        self.config = LatentEuclidConfig()
        self.config.hidden_size = target_dim

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None):
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
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        
        last_hidden_states = outputs.hidden_states[-1] # [batch, seq_len, hidden_size]
        
        batch_size = input_ids.shape[0]
        max_N = 0
        all_thought_positions = []
        
        for b in range(batch_size):
            b_input_ids = input_ids[b]
            
            # Fast vectorized dimensional matching against all pre-cached <thought_X> dynamic structures
            # Because the inputs were explicitly tokenized sequentially from the generated string: `<thought_1><thought_2>...<HALT>`
            # These nonzeros inherently trace perfectly chronologically to the arbitrary topology sequence.
            is_thought = torch.isin(b_input_ids, torch.tensor(self.thought_ids, device=b_input_ids.device))
            pos = is_thought.nonzero(as_tuple=True)[0]
            
            max_N = max(max_N, len(pos))
            all_thought_positions.append(pos)
            
        # Padded array dynamically spanning to the longest latent unroll execution within the current causal block
        predicted_latents_padded = torch.zeros(batch_size, max_N, last_hidden_states.size(-1), device=last_hidden_states.device, dtype=last_hidden_states.dtype)
        
        for b in range(batch_size):
            pos = all_thought_positions[b]
            if len(pos) > 0:
                thought_vectors = last_hidden_states[b][pos]
                predicted_latents_padded[b, :len(pos), :] = thought_vectors
        
        # Shape: [batch, max_N, target_dim]
        projected_latents = self.predictor(predicted_latents_padded)
        
        return projected_latents

    def forward_contrastive_texts(
        self,
        cod_texts: list,
        cot_texts: list,
        chunk_size: int | None = None,
    ):
        """
        Encode CoD and CoT text sequences into fixed-size embeddings for contrastive learning.

        Appends <REASON> to every string, runs text-only forward passes through the VLM,
        and extracts the hidden state at the <REASON> position as a pooled representation.
        Projects through the predictor head so both streams live in the same target latent space.

        Args:
            chunk_size: If set, processes texts in mini-batches of this size and concatenates
                        the results. Reduces peak GPU memory when encoding large batches
                        (e.g. the sanity check's 32 samples) at the cost of more sequential
                        VLM calls. Set to the training batch_size to stay within budget.
                        None (default) processes all texts in a single forward pass.

        Returns:
            Z_cod: (Batch, target_dim)
            Z_cot: (Batch, target_dim)
        """
        device = next(self.predictor.parameters()).device
        reason_str = " <REASON>"

        cod_inputs_raw = [t + reason_str for t in cod_texts]
        cot_inputs_raw = [t + reason_str for t in cot_texts]

        def _encode_chunk(texts: list):
            encodings = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = self.vlm(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden_size]

            reason_id = self.reason_token_id
            b_size = encodings.input_ids.shape[0]
            pooled = []
            for b in range(b_size):
                positions = (encodings.input_ids[b] == reason_id).nonzero(as_tuple=True)[0]
                # Use the last occurrence; fall back to final token if sentinel is missing
                pos = int(positions[-1]) if len(positions) > 0 else -1
                pooled.append(last_hidden[b, pos])

            pooled_tensor = torch.stack(pooled, dim=0)  # [B, hidden_size]
            return self.predictor(pooled_tensor)         # [B, target_dim]

        def _encode(texts: list):
            if chunk_size is None or len(texts) <= chunk_size:
                return _encode_chunk(texts)
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            return torch.cat([_encode_chunk(chunk) for chunk in chunks], dim=0)

        Z_cod = _encode(cod_inputs_raw)
        Z_cot = _encode(cot_inputs_raw)

        return Z_cod, Z_cot

if __name__ == "__main__":
    # Test Scaffold
    # model = LatentEuclid()
    pass
