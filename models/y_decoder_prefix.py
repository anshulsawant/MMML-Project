'''
LatentEuclid: The Y-Decoder Continuous Generative Loop

Because pure transformer decoder-only LLMs lack cross-attention layers, mapping 
predicted target latents natively into generation relies strictly upon "Soft Prompt Prefix Tuning."

This component accepts the continuous `K` projected thought manifolds from the `X-Encoder`
and appends them logically before the standard generation inputs. This allows the frozen
0.6B baseline syntax to inherently leverage visual topologies to "calculate" text probabilities 
via autoregressive extraction.
'''

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoConfig

class YDecoderPrefix(nn.Module):
    """
    Wraps frozen Qwen3-0.6B to accept the 4 predicted `thought` vectors 
    as Soft Prompts directly appended via Prefix Tuning mechanisms.
    """
    def __init__(self, 
                 target_model_id: str = "Qwen/Qwen3-0.6B", 
                 k_steps: int = 4):
        super().__init__()
        
        self.k_steps = k_steps
        
        # Load Frozen Decoder
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model config to check for VL architecture
        config = AutoConfig.from_pretrained(target_model_id, trust_remote_code=True)
        
        # Load the base model correctly depending on architecture
        if "VL" in config.model_type.upper() or "VL" in target_model_id.upper():
            self.decoder = AutoModelForImageTextToText.from_pretrained(
                target_model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else None
            )
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                target_model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else None
            )

        # Freeze the entire underlying LLM/VLM
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        self.embedding_dim = getattr(self.decoder.config, "hidden_size", getattr(getattr(self.decoder.config, "text_config", None), "hidden_size", None))
        
        # In our pipeline, the X-Encoder predictor outputs strictly to self.embedding_dim.
        # This 2-layer MLP maps the structural non-linearities between the modalities.
        self.prefix_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2).to(torch.bfloat16),
            nn.GELU().to(torch.bfloat16),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim).to(torch.bfloat16)
        )

    def forward(self, predicted_latents, text_prompts=None, labels=None):
        """
        predicted_latents: [batch, K, dim] from LatentEuclid.
        text_prompts: Optional List[str] guiding the generation (e.g. "The final answer is: ")
        labels: Optional List[str] of target answers for computing cross-entropy loss.
        
        Returns CausalLMOutputWithPast (includes logits and loss if labels are provided).
        """
        device = self.decoder.device
        predicted_latents = predicted_latents.to(device)
        
        # 1. Project the latents to the exact embedding dimension
        # Shape: [batch, K, embedding_dim]
        soft_prefixes = self.prefix_projection(predicted_latents)
        
        # 2. Get embed weights for the text tokens
        if text_prompts is None:
            # Default generation prompt
            text_prompts = [""] * soft_prefixes.shape[0]
            
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Extract the native continuous embeddings for the text
        # Shape: [batch, seq_len, embedding_dim]
        text_embeddings = self.decoder.get_input_embeddings()(input_ids)
        
        # 3. Concatenate: [Text Embeddings (Question)] + [Soft Prefixes]
        # Shape: [batch, seq_len + K, embedding_dim]
        inputs_embeds = torch.cat([text_embeddings, soft_prefixes], dim=1)
        
        # 4. Expand the attention mask to cover the K soft prompt tokens appended *after* the text
        # Shape: [batch, K] of 1s
        prefix_mask = torch.ones(
            (soft_prefixes.shape[0], self.k_steps), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        # Shape: [batch, seq_len + K]
        extended_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        
        # 5. Handle Target Labels for Training (Phase 4.5)
        extended_labels = None
        if labels is not None:
            # Tokenize the target text separately
            label_inputs = self.tokenizer(labels, return_tensors="pt", padding=True).to(device)
            label_embeddings = self.decoder.get_input_embeddings()(label_inputs.input_ids)
            
            # Combine everything: [Text Prompts] + [Soft Prefixes] + [Target Labels]
            inputs_embeds = torch.cat([text_embeddings, soft_prefixes, label_embeddings], dim=1)
            
            # Update attention mask
            extended_attention_mask = torch.cat([attention_mask, prefix_mask, label_inputs.attention_mask], dim=1)
            
            # -100 is the standard PyTorch ignore_index for CrossEntropyLoss
            # We must ignore the K continuous prefix latents AND the text prompts (e.g. "Question: Find x...")
            ignore_prompts = torch.full_like(input_ids, -100)
            ignore_prefix = torch.full((soft_prefixes.shape[0], self.k_steps), -100, dtype=torch.long, device=device)
            
            # Mask out padding tokens in the target labels so we only compute CE loss on the actual answer bytes
            masked_label_ids = label_inputs.input_ids.clone()
            masked_label_ids[label_inputs.attention_mask == 0] = -100
            
            # Only the final target answers factor into the loss calculation
            extended_labels = torch.cat([ignore_prompts, ignore_prefix, masked_label_ids], dim=1)

            # --- TEMPORARY DEBUG BLOCK ---
            if not hasattr(self, '_debug_printed'):
                print("\n=== DEBUG: Token Leakage Check ===")
                try:
                    # Decode the raw input prompt text tokens
                    prompt_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    print(f"RAW PROMPT TOKENS: {prompt_str}")
                    
                    # Decode the raw answer text tokens
                    label_str = self.tokenizer.decode(label_inputs.input_ids[0], skip_special_tokens=False)
                    print(f"RAW LABEL TOKENS: {label_str}")
                    
                    # Print the exact shape of the vectors
                    print(f"Prefix Shape: {soft_prefixes.shape} | Prompt Shape: {text_embeddings.shape} | Label Shape: {label_embeddings.shape}")
                    print(f"Extended Labels Tensor: {extended_labels[0].tolist()}")
                except Exception as e:
                    print(f"Debug print failed: {e}")
                print("==================================\n")
                self._debug_printed = True

        # 6. Forward pass through frozen LLM using `inputs_embeds` instead of integer IDs
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            output_hidden_states=False,
            return_dict=True
        )
        
        return outputs

    @torch.no_grad()
    def generate(self, predicted_latents, text_prompts=None, max_new_tokens=20):
        """
        Inference-time evaluation to predict the exact short string (e.g., \boxed{40}).
        """
        device = self.decoder.device
        soft_prefixes = self.prefix_projection(predicted_latents.to(device))
        
        if text_prompts is None:
            text_prompts = [""] * soft_prefixes.shape[0]
            
        # Switch padding to left for batched generation, then restore
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True).to(device)
        self.tokenizer.padding_side = original_padding_side
        
        text_embeddings = self.decoder.get_input_embeddings()(inputs.input_ids)
        inputs_embeds = torch.cat([text_embeddings, soft_prefixes], dim=1)
        
        prefix_mask = torch.ones((soft_prefixes.shape[0], self.k_steps), dtype=inputs.attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([inputs.attention_mask, prefix_mask], dim=1)
        
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=[self.tokenizer.eos_token_id, 151645], # 151645 = <|im_end|>
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )
        
        # Decode the output tokens back to strings
        decoded_answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_answers

if __name__ == "__main__":
    # Test Scaffold
    # decoder = YDecoderPrefix()
    pass
