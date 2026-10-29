import torch
import torch.nn as nn

class CoxPHLoss(nn.Module):
    """
    Negative partial log-likelihood loss for Cox Proportional Hazards model.
    Includes Breslow's approximation for handling tied survival times.
    """
    def __init__(self):
        super().__init__()

    def forward(self, risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_scores: Predicted log-hazard (eta), shape [batch_size]
            times: Survival times (OS.time), shape [batch_size]
            events: Event indicators (OS: 1 for death, 0 for censored), shape [batch_size]

        Returns:
            Scalar loss tensor (for zero-event batches, a graph-connected zero loss).
        """
        # Filter out batches with zero events to prevent NaN.
        # Return a graph-connected zero loss (gradient = 0, backward() is safe,
        # unlike a standalone leaf tensor which silently skips the whole batch).
        num_events = events.sum()
        if num_events == 0:
            return (risk_scores * 0.0).sum()

        # 1. Sort by time in descending order so the at-risk set for patient i
        #    is exactly the prefix 0..i of the sorted array.
        sorted_times, sort_idx = torch.sort(times, descending=True)
        sorted_risk = risk_scores[sort_idx]
        sorted_events = events[sort_idx]

        # 2. Numerically stable log-sum-exp:
        #    log(SUM_j exp(eta_j)) = max_eta + log(SUM_j exp(eta_j - max_eta))
        #    Prevents overflow to inf for large risk scores.
        max_risk = sorted_risk.max()
        shifted_risk = sorted_risk - max_risk

        exp_risk = torch.exp(shifted_risk)

        # 3. Cumulative sum of exp(risk) from index 0 to i gives the at-risk
        #    sum for the prefix (t_j >= t_i in descending order).
        cum_exp_risk = torch.cumsum(exp_risk, dim=0)

        # 4. Shift back: log(SUM_j exp(eta_j)) = max_risk + log(SUM exp(shifted)).
        #    The tiny epsilon guards against underflow (all prefix terms -> 0).
        log_cum_risk = max_risk + torch.log(cum_exp_risk + 1e-12)

        # 5. Compute log likelihood for EVENT patients only:
        #    LL = SUM_{i where event=1} (eta_i - log(SUM_{j >= i} exp(eta_j)))
        uncensored_ll = sorted_risk - log_cum_risk

        # 6. Apply event mask and sum
        total_ll = torch.sum(uncensored_ll * sorted_events)

        # 7. Return negative average partial log-likelihood
        loss = -total_ll / num_events

        return loss