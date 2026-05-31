import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLossFactory(nn.Module):
    def __init__(self, 
                 loss_type: str = "vicreg", 
                 sim_threshold: float = 0.90,
                 vicreg_sim_coeff: float = 25.0,
                 vicreg_var_coeff: float = 25.0,
                 vicreg_cov_coeff: float = 1.0,
                 temperature: float = 0.07,
                 gamma: float = 1.0,
                 queue_size: int = 128,
                 hidden_dim: int = 3584):
        super().__init__()
        valid_types = ["info_nce_vanilla", "info_nce_threshold", "vicreg", "huber_cosine"]
        if loss_type not in valid_types:
            raise ValueError(f"Unknown loss_type. Must be one of {valid_types}")
            
        self.loss_type = loss_type
        self.sim_threshold = sim_threshold
        self.temperature = temperature
        self.gamma = gamma
        self.queue_size = queue_size
        self.hidden_dim = hidden_dim
        
        # VICReg specific hyperparameters
        self.sim_coeff = vicreg_sim_coeff
        self.var_coeff = vicreg_var_coeff
        self.cov_coeff = vicreg_cov_coeff

        # Learnable temperature for the InfoNCE contrastive loss (CLIP-style)
        # Initialised to ln(1/0.07) ≈ 2.659 so exp(logit_scale) ≈ 1/0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

        # MoCo-style FIFO queues for negatives from historical batches.
        queue_cod = F.normalize(torch.randn(queue_size, hidden_dim), dim=-1)
        queue_cot = F.normalize(torch.randn(queue_size, hidden_dim), dim=-1)
        self.register_buffer("queue_cod", queue_cod)
        self.register_buffer("queue_cot", queue_cot)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # When False, compute_contrastive_loss falls back to pure in-batch
        # symmetric InfoNCE and skips queue updates. Useful for sanity checks
        # where the queue contains only random noise.
        self.use_queue = True
        
    def forward(self, predicted, targets, Z_cod=None, Z_cot=None):
        """
        predicted: [batch, K, dim] - Output from LatentEuclid Predictor
        targets:   [batch, K, dim] - Target vectors from frozen Qwen-0.5B
        Z_cod:     [batch, dim]    - Pooled CoD embeddings (optional)
        Z_cot:     [batch, dim]    - Pooled CoT embeddings (optional)
        """
        batch, k_steps, dim = predicted.shape
        total_loss = 0.0
        metrics = {
            "loss/vicreg_total": 0.0,
            "loss/invariance_cos": 0.0,
            "loss/variance_loss": 0.0,
            "loss/variance_std_physical": 0.0,
            "loss/covariance_cor": 0.0,
            "loss/info_nce": 0.0,
            "loss/contrastive": 0.0,
        }
        
        # We compute the loss iteratively across the sequence of K steps
        for k in range(k_steps):
            pred_k = predicted[:, k, :]
            targ_k = targets[:, k, :]
            
            if self.loss_type == "info_nce_vanilla":
                loss_val = self.compute_info_nce(pred_k, targ_k, use_threshold=False)
                total_loss += loss_val
                metrics["loss/info_nce"] += loss_val.item()
            elif self.loss_type == "info_nce_threshold":
                loss_val = self.compute_info_nce(pred_k, targ_k, use_threshold=True)
                total_loss += loss_val
                metrics["loss/info_nce"] += loss_val.item()
            elif self.loss_type == "vicreg":
                loss_val, vicreg_metrics = self.compute_vicreg(pred_k, targ_k)
                total_loss += loss_val
                metrics["loss/vicreg_total"] += loss_val.item()
                metrics["loss/invariance_cos"] += vicreg_metrics["invariance_cos"]
                metrics["loss/variance_loss"] += vicreg_metrics["variance_loss"]
                metrics["loss/variance_std_physical"] += vicreg_metrics["variance_std_physical"]
                metrics["loss/covariance_cor"] += vicreg_metrics["covariance_cor"]
            elif self.loss_type == "huber_cosine":
                loss_val, hc_metrics = self.compute_huber_cosine(pred_k, targ_k)
                total_loss += loss_val
                if "loss/huber_cosine_total" not in metrics:
                    metrics["loss/huber_cosine_total"] = 0.0
                    metrics["loss/cosine_angular"] = 0.0
                    metrics["loss/huber_magnitude"] = 0.0
                metrics["loss/huber_cosine_total"] += loss_val.item()
                metrics["loss/cosine_angular"] += hc_metrics["cosine_angular"]
                metrics["loss/huber_magnitude"] += hc_metrics["huber_magnitude"]
                
        # Average loss and metrics over the K reasoning steps
        total_loss = total_loss / k_steps
        for k in metrics.keys():
            metrics[k] /= k_steps

        # Optional dual-stream contrastive loss (CoD vs CoT)
        if Z_cod is not None and Z_cot is not None:
            contrastive_loss = self.compute_contrastive_loss(Z_cod, Z_cot)
            total_loss = total_loss + self.gamma * contrastive_loss
            metrics["loss/contrastive"] = contrastive_loss.item()

        return total_loss, metrics

    @torch.no_grad()
    def _dequeue_and_enqueue(self, Z_cod, Z_cot):
        """
        Push current batch embeddings into the FIFO queue and advance pointer.
        """
        Z_cod = Z_cod.detach()
        Z_cot = Z_cot.detach()

        batch_size = Z_cod.shape[0]
        if batch_size == 0:
            return

        # If a batch is larger than queue, keep only the most recent queue_size entries.
        if batch_size >= self.queue_size:
            Z_cod = Z_cod[-self.queue_size:]
            Z_cot = Z_cot[-self.queue_size:]
            batch_size = self.queue_size

        ptr = int(self.queue_ptr.item())
        end_ptr = ptr + batch_size

        if end_ptr <= self.queue_size:
            self.queue_cod[ptr:end_ptr] = Z_cod
            self.queue_cot[ptr:end_ptr] = Z_cot
        else:
            first_chunk = self.queue_size - ptr
            second_chunk = batch_size - first_chunk
            self.queue_cod[ptr:] = Z_cod[:first_chunk]
            self.queue_cot[ptr:] = Z_cot[:first_chunk]
            self.queue_cod[:second_chunk] = Z_cod[first_chunk:]
            self.queue_cot[:second_chunk] = Z_cot[first_chunk:]

        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.queue_size

    def compute_contrastive_loss(self, Z_cod, Z_cot):
        """
        Symmetric InfoNCE with a MoCo-style queue of historical negatives.

        Positives are in-batch CoD-CoT pairs.
        Negatives come from queue_cod/queue_cot.
        """
        Z_cod = F.normalize(Z_cod, dim=-1)
        Z_cot = F.normalize(Z_cot, dim=-1)

        if Z_cod.shape[-1] != self.hidden_dim or Z_cot.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Embedding dim mismatch. Expected hidden_dim={self.hidden_dim}, "
                f"got Z_cod={Z_cod.shape[-1]}, Z_cot={Z_cot.shape[-1]}"
            )

        # Clamp logit_scale to ln(100) ≈ 4.605 so the temperature stays in
        # [1/100, ...] and never drives logits into a regime where gradients vanish.
        scale = self.logit_scale.clamp(max=4.6052).exp()

        if not self.use_queue:
            # In-batch symmetric InfoNCE (no queue negatives).
            # Safe to run even when B=2; queue is frozen and untouched.
            logits = scale * (Z_cod @ Z_cot.T)  # [B, B]
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
            return loss

        # Positive logits: [B, 1]
        pos_logits = scale * torch.sum(Z_cod * Z_cot, dim=-1, keepdim=True)

        # Negative logits from the queue: [B, Q]
        neg_logits_cod = scale * (Z_cod @ self.queue_cot.T)
        neg_logits_cot = scale * (Z_cot @ self.queue_cod.T)

        # Combined logits: positive always at index 0.
        logits_cod = torch.cat([pos_logits, neg_logits_cod], dim=1)
        logits_cot = torch.cat([pos_logits, neg_logits_cot], dim=1)

        labels = torch.zeros(logits_cod.shape[0], dtype=torch.long, device=logits_cod.device)
        loss_cod = F.cross_entropy(logits_cod, labels)
        loss_cot = F.cross_entropy(logits_cot, labels)
        loss = (loss_cod + loss_cot) / 2.0

        # Update queue after computing the loss so current batch is used in future steps.
        self._dequeue_and_enqueue(Z_cod, Z_cot)

        return loss

    def compute_info_nce(self, pred, targ, use_threshold=False):
        """
        Standard InfoNCE: Align pred with targ while pushing away from other targs in the batch.
        """
        # Normalize representations
        pred = F.normalize(pred, dim=1)
        targ = F.normalize(targ, dim=1)
        
        # Similarity matrix: [batch_pred, batch_targ]
        sim_matrix = torch.matmul(pred, targ.T) / self.temperature
        
        # Labels are the diagonal (Pred i matches Targ i)
        labels = torch.arange(pred.shape[0], device=pred.device)
        
        if use_threshold:
            # Mask out false negatives where Target i is extremely similar to Target j
            with torch.no_grad():
                target_sims = torch.matmul(targ, targ.T) # [batch, batch] target similarity
                # Find indices where sim > threshold, excluding self (diagonal)
                eye = torch.eye(target_sims.shape[0], device=targ.device).bool()
                false_neg_mask = (target_sims > self.sim_threshold) & (~eye)
                
            # We set the similarity score of false negatives to -infinity so they drop out of softmax
            sim_matrix = sim_matrix.masked_fill(false_neg_mask, -1e9)
            
        return F.cross_entropy(sim_matrix, labels)

    def compute_huber_cosine(self, x, y):
        """
        Supervised Continuous Alignment Loss.
        Because Target Y is a frozen LLM, we strictly enforce both angular 
        direction (Cosine) and scale magnitude (Huber).
        """
        # Directional alignment: Target 1 means "make these vectors point the same way"
        target_ones = torch.ones(x.shape[0], device=x.device)
        cos_loss = F.cosine_embedding_loss(x, y, target_ones)
        
        # Magnitude/Scale alignment: Smooth L1 treats small errors as L2 and large errors as L1
        huber_loss = F.smooth_l1_loss(x, y, beta=1.0)
        
        # Balance coefficients (magnitude loss is typically slightly larger natively)
        total_loss = (10.0 * cos_loss) + huber_loss
        
        metrics = {
            "cosine_angular": cos_loss.item(),
            "huber_magnitude": huber_loss.item()
        }
        
        return total_loss, metrics

    def compute_vicreg(self, x, y):
        """
        VICReg: Variance-Invariance-Covariance Regularization.
        x: Predicted representations
        y: Target representations or Augmented view representations
        """
        # Invariance (Similarity): Cosine Distance between X and Y
        sim_loss = 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
        
        # Center representations
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        # Variance: Push standard deviation of each feature dimension towards 1 (gamma=1)
        std_x = torch.sqrt(x.var(dim=0) + 1e-04)
        std_y = torch.sqrt(y.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - std_x))
        
        # Covariance: Decorrelate distinct features
        batch_size = x.shape[0]
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        
        # Sum of off-diagonal squared elements
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(x.shape[1]) + \
                   self.off_diagonal(cov_y).pow_(2).sum().div(y.shape[1])
                   
        total_vicreg_loss = (self.sim_coeff * sim_loss) + (self.var_coeff * var_loss) + (self.cov_coeff * cov_loss)
        metrics = {
            "invariance_cos": sim_loss.item(),
            "variance_loss": var_loss.item(),
            "variance_std_physical": torch.mean(std_x).item(),
            "covariance_cor": cov_loss.item()
        }
        
        return total_vicreg_loss, metrics
        
    @staticmethod
    def off_diagonal(x):
        """Helper function to get off-diagonal elements of a square matrix"""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if __name__ == "__main__":
    # Scaffold testing
    # factory = AlignmentLossFactory(loss_type="vicreg")
    pass
