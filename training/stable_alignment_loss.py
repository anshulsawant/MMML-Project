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
                 temperature: float = 0.07):
        super().__init__()
        valid_types = ["info_nce_vanilla", "info_nce_threshold", "vicreg"]
        if loss_type not in valid_types:
            raise ValueError(f"Unknown loss_type. Must be one of {valid_types}")
            
        self.loss_type = loss_type
        self.sim_threshold = sim_threshold
        self.temperature = temperature
        
        # VICReg specific hyperparameters
        self.sim_coeff = vicreg_sim_coeff
        self.var_coeff = vicreg_var_coeff
        self.cov_coeff = vicreg_cov_coeff
        
    def forward(self, predicted, targets):
        """
        predicted: [batch, K, dim] - Output from LatentEuclid Predictor
        targets: [batch, K, dim] - Target vectors from frozen Qwen-0.5B
        """
        batch, k_steps, dim = predicted.shape
        total_loss = 0.0
        metrics = {
            "loss/vicreg_total": 0.0,
            "loss/invariance_mse": 0.0,
            "loss/variance_std": 0.0,
            "loss/covariance_cor": 0.0,
            "loss/info_nce": 0.0
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
                metrics["loss/invariance_mse"] += vicreg_metrics["invariance_mse"]
                metrics["loss/variance_std"] += vicreg_metrics["variance_std"]
                metrics["loss/covariance_cor"] += vicreg_metrics["covariance_cor"]
                
        # Average loss and metrics over the K reasoning steps
        total_loss = total_loss / k_steps
        for k in metrics.keys():
            metrics[k] /= k_steps
            
        return total_loss, metrics

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
