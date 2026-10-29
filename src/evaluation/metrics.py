from lifelines.utils import concordance_index
import numpy as np
import torch

def compute_c_index(risk_scores: np.ndarray, event_times: np.ndarray, event_observed: np.ndarray) -> float:
    """
    Computes the Concordance Index (C-index) for survival models.
    
    Args:
        risk_scores: Predicted risk scores (higher means higher hazard)
        event_times: Observed survival times
        event_observed: Event indicator (1 for death, 0 for censored)
        
    Returns:
        C-index score between 0.0 and 1.0. 
        A score of 0.5 is random chance, 1.0 is perfect concordance.
    """
    # Note: lifelines concordance_index expects standard predictions (higher survival time),
    # but since our risk scores mean *lower* survival time, we invert them by negating them.
    # Alternatively, lifelines can handle risk scores directly if we pass them as -risk_scores
    
    return concordance_index(
        event_times=event_times,
        predicted_scores=-risk_scores,
        event_observed=event_observed
    )
