import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class EuclidGRPOConfig(PretrainedConfig):
    model_type = "euclid_grpo"

class UnifiedEuclidGRPO(PreTrainedModel):
    """
    Wraps the X-Encoder (LatentEuclid) and Y-Decoder (YDecoderPrefix)
    into a mathematically singular Hugging Face model supporting `.generate()` and `.forward()`.
    This is required natively by `trl.GRPOTrainer`.
    """
    config_class = EuclidGRPOConfig
    
    def __init__(self, config, x_encoder, y_decoder):
        super().__init__(config)
        self.x_encoder = x_encoder
        self.y_decoder = y_decoder
        
        # Mirror generation config to target decoder
        self.generation_config = y_decoder.decoder.generation_config
        
    def _extract_latents(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids):
        """
        Dynamically generates the K continuous topological latents using the X-Encoder.
        """
        # For GRPO, we can use a static number of steps (e.g. 4) or 
        # dynamically track halts internally exactly like eval_e2e.py!
        # Simplified to K=4 strictly for stable GRPO rollouts unless specified otherwise.
        # This mirrors the continuous projection step.
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # NOTE: Ideally this pulls from config.
        max_steps = 4
        
        dynamic_input_ids = input_ids
        dynamic_attention_mask = attention_mask
        dynamic_mm = mm_token_type_ids
        
        # Generate latents exactly like training/eval scripts
        for step in range(max_steps):
            t_id = torch.tensor([[self.x_encoder.thought_ids[step]]] * batch_size).to(device)
            dynamic_input_ids = torch.cat([dynamic_input_ids, t_id], dim=1)
            
            new_attn = torch.ones((batch_size, 1), dtype=dynamic_attention_mask.dtype, device=device)
            dynamic_attention_mask = torch.cat([dynamic_attention_mask, new_attn], dim=1)
            
            if dynamic_mm is not None:
                new_mm_token = torch.zeros((batch_size, 1), dtype=dynamic_mm.dtype, device=device)
                dynamic_mm = torch.cat([dynamic_mm, new_mm_token], dim=1)
                
            predicted_latents = self.x_encoder(
                input_ids=dynamic_input_ids, 
                pixel_values=pixel_values, 
                attention_mask=dynamic_attention_mask,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=dynamic_mm
            )
            
        return predicted_latents

    def generate(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None, **kwargs):
        """
        Drives the policy generation trajectory. 
        """
        predicted_latents = self._extract_latents(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
        
        # We format clean prompts (removing thought tags since our latents replaced them visually)
        # But wait! The input_ids might already be tokenized natively by Trl!
        # We circumvent by decoding the inputs natively before handing them to y_decoder.generate
        text_prompts = self.y_decoder.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        return self.y_decoder.generate(
            predicted_latents=predicted_latents, 
            text_prompts=text_prompts,
            **kwargs
        )
        
    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None, labels=None, **kwargs):
        """
        Computes forward-pass teacher-forcing log probabilities for PPO/GRPO updates.
        """
        predicted_latents = self._extract_latents(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
        
        text_prompts = self.y_decoder.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Use our prefix wrapper directly, it naturally computes Cross-Entropy if labels are provided!
        return self.y_decoder(
            predicted_latents=predicted_latents,
            text_prompts=text_prompts,
            labels=labels
        )
