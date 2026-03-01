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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.decoder = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Freeze the base LLM weights entirely
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        self.embedding_dim = getattr(self.decoder.config, "hidden_size", getattr(getattr(self.decoder.config, "text_config", None), "hidden_size", None))
        
        # In our pipeline, the X-Encoder predictor outputs strictly to self.embedding_dim.
        # This linear layer can map any structural mismatch if we experiment with other models.
        self.prefix_projection = nn.Linear(self.embedding_dim, self.embedding_dim).to(torch.bfloat16)

    def forward(self, predicted_latents, text_prompts=None):
        """
        predicted_latents: [batch, K, dim] from LatentEuclid.
        text_prompts: Optional List[str] guiding the generation (e.g. "The final answer is: ")
        
        Returns logits from the autoregressive generation.
        """
        device = self.decoder.device
        predicted_latents = predicted_latents.to(device)
        
        # 1. Project the latents to the exact embedding dimension
        # Shape: [batch, K, embedding_dim]
        soft_prefixes = self.prefix_projection(predicted_latents)
        
        # 2. Get embed weights for the text tokens
        if text_prompts is None:
            # Default generation prompt
            text_prompts = ["Answer: "] * soft_prefixes.shape[0]
            
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Extract the native continuous embeddings for the text
        # Shape: [batch, seq_len, embedding_dim]
        text_embeddings = self.decoder.get_input_embeddings()(input_ids)
        
        # 3. Concatenate: [Soft Prefixes] + [Text Embeddings]
        # Shape: [batch, K + seq_len, embedding_dim]
        inputs_embeds = torch.cat([soft_prefixes, text_embeddings], dim=1)
        
        # 4. Expand the attention mask to cover the K soft prompt tokens
        # Shape: [batch, K] of 1s
        prefix_mask = torch.ones(
            (soft_prefixes.shape[0], self.k_steps), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        # Shape: [batch, K + seq_len]
        extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 5. Forward pass through frozen LLM using `inputs_embeds` instead of integer IDs
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        return outputs.logits

    @torch.no_grad()
    def generate(self, predicted_latents, text_prompts=None, max_new_tokens=20):
        """
        Inference-time evaluation to predict the exact short string (e.g., \boxed{40}).
        """
        device = self.decoder.device
        soft_prefixes = self.prefix_projection(predicted_latents.to(device))
        
        if text_prompts is None:
            text_prompts = ["Answer: "] * soft_prefixes.shape[0]
            
        inputs = self.tokenizer(text_prompts, return_tensors="pt", padding=True).to(device)
        
        text_embeddings = self.decoder.get_input_embeddings()(inputs.input_ids)
        inputs_embeds = torch.cat([soft_prefixes, text_embeddings], dim=1)
        
        prefix_mask = torch.ones((soft_prefixes.shape[0], self.k_steps), dtype=inputs.attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)
        
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the output tokens back to strings
        decoded_answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_answers

if __name__ == "__main__":
    # Test Scaffold
    # decoder = YDecoderPrefix()
    pass
