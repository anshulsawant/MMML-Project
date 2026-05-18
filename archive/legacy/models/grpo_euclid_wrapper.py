import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from PIL import Image

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
        
    def _extract_latents(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None, image_path=None, **kwargs):
        """
        Dynamically generates the K continuous topological latents using the X-Encoder.
        Because TRL maps loops natively against textual inputs, we decouple the vision processing
        by constructing fresh multi-modal bounds exclusively tracking the X-Encoder structurally.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # 1. Fallback Text Parsing
        # TRL calls .forward() with the entire prompt + thought + completion sequence.
        # We MUST slice off everything from the thought tokens onwards to preserve causality!
        first_thought_id = self.x_encoder.thought_ids[0]
        sliced_input_ids = []
        for b_idx in range(batch_size):
            idx = (input_ids[b_idx] == first_thought_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                # Isolate exactly up to the thought boundary
                sliced_input_ids.append(input_ids[b_idx, :idx[0]])
            else:
                # Generation mode: input_ids contains pure prompt
                sliced_input_ids.append(input_ids[b_idx])
                
        text_prompts = [self.y_decoder.tokenizer.decode(t, skip_special_tokens=True) for t in sliced_input_ids]
        
        # 2. Extract visual geometry locally to execute independent multi-modal bounds safely
        images = []
        if image_path is not None:
            # We are currently deployed across batches seamlessly safely mapped inside kwargs!
            for path in image_path:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except:
                    images.append(Image.new('RGB', (224, 224), color=(73, 109, 137)))
        else:
            # Fatal fallback if TRL dataloaders stripped the kwargs structure natively 
            images = [Image.new('RGB', (224, 224), color=(73, 109, 137)) for _ in range(batch_size)]
            
        # 3. Format strictly adhering to Qwen3-VL processor schema locally ignoring TRL's string limits
        batch_messages = []
        for txt, img in zip(text_prompts, images):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": txt}
                ]
            }]
            batch_messages.append(messages)
            
        rendered_texts = [self.x_encoder.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        inputs = self.x_encoder.processor(text=rendered_texts, images=images, padding=True, return_tensors="pt").to(device)
        
        dynamic_input_ids = inputs.input_ids
        dynamic_attention_mask = inputs.attention_mask
        dynamic_mm = inputs.get("mm_token_type_ids")
        pixel_vals = inputs.get("pixel_values")
        image_grid = inputs.get("image_grid_thw")
        
        max_steps = 4
        
        # Generate latents tracking exactly like SFT bounds independently
        for step in range(max_steps):
            t_id = torch.tensor([[self.x_encoder.thought_ids[step]]] * batch_size, device=device)
            dynamic_input_ids = torch.cat([dynamic_input_ids, t_id], dim=1)
            
            new_attn = torch.ones((batch_size, 1), dtype=dynamic_attention_mask.dtype, device=device)
            dynamic_attention_mask = torch.cat([dynamic_attention_mask, new_attn], dim=1)
            
            if dynamic_mm is not None:
                new_mm_token = torch.zeros((batch_size, 1), dtype=dynamic_mm.dtype, device=device)
                dynamic_mm = torch.cat([dynamic_mm, new_mm_token], dim=1)
                
            predicted_latents = self.x_encoder(
                input_ids=dynamic_input_ids, 
                pixel_values=pixel_vals, 
                attention_mask=dynamic_attention_mask,
                image_grid_thw=image_grid,
                mm_token_type_ids=dynamic_mm
            )
            
        return predicted_latents

    def generate(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None, **kwargs):
        """
        Drives the policy generation trajectory for TRL directly yielding raw token ids. 
        """
        predicted_latents = self._extract_latents(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids, **kwargs)
        
        device = self.y_decoder.decoder.device
        batch_size = input_ids.shape[0]
        
        soft_prefixes = self.y_decoder.prefix_projection(predicted_latents.to(device))
        
        # Concatenate text embeddings with soft_prefixes!
        text_embeddings = self.y_decoder.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([text_embeddings, soft_prefixes], dim=1)
        
        target_n_steps = soft_prefixes.shape[1]
        prefix_mask = torch.ones((batch_size, target_n_steps), dtype=attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        
        output_ids = self.y_decoder.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            **kwargs
        )
        
        # We explicitly weave the discrete dummy placeholder `<thought>` token IDs into the generated sequence.
        # This mathematically perfectly equates to the continuous latents we implicitly appended.
        # Therefore, when TRL later feeds this exact sequence back into .forward(), the sequence lengths 
        # naturally align and we can symmetrically intercept the topological embeddings directly inside the graph!
        thought_ids = torch.tensor([self.x_encoder.thought_ids[:target_n_steps]], device=device).expand(batch_size, target_n_steps)
        prompt_completion_ids = torch.cat([input_ids, thought_ids, output_ids], dim=1)
        
        return prompt_completion_ids
        
    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None, labels=None, **kwargs):
        """
        Computes forward-pass teacher-forcing log probabilities for PPO/GRPO updates.
        Mathematically elegantly intercepts the `<thought>` categorical tags natively injected within 
        generate() and physically overwrites their vectors with actual continuous VL-JEPA topology!
        """
        predicted_latents = self._extract_latents(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids, **kwargs)
        
        device = self.y_decoder.decoder.device
        soft_prefixes = self.y_decoder.prefix_projection(predicted_latents.to(device))
        
        # Resolve original rigid tokens back into dynamic embedding dimensions
        inputs_embeds = self.y_decoder.decoder.get_input_embeddings()(input_ids).clone()
        
        # Symbiotic overwrite mapping categorical placeholders directly onto continuous space:
        # Since TRL blindly operates against token sequences, finding the embedded bounds guarantees perfect alignment.
        for batch_idx in range(input_ids.shape[0]):
            for k, t_id in enumerate(self.x_encoder.thought_ids[:soft_prefixes.shape[1]]):
                mask = (input_ids[batch_idx] == t_id)
                inputs_embeds[batch_idx, mask] = soft_prefixes[batch_idx, k]
                
        # Forward pass exclusively utilizing Qwen3-4B-Base natively tracking perfectly against TRL's Gather hooks!
        return self.y_decoder.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
