import torch

def compute_classification_loss(out_cf_class, target_class, device):
    """Compute the loss between the predicted class of the counterfactual and the target class.

    Args:
        out_cf_class: Classifier predictions on counterfactuals
        target_class: Desired target class for counterfactuals
        device: CPU or GPU

    Returns:
        dist_pred_target: Classification loss
    """
    target_tensor = torch.full((out_cf_class.shape), target_class).to(device)
    dist_pred_target = torch.sqrt(torch.pow((out_cf_class - target_tensor), 2)).to(device) 
    return dist_pred_target

def compute_similarity_loss(cf, X_query=[]):
    """Penalizes large differences between query and counterfactual (L1 loss).

    Args:
        cf: counterfactual
        X_query: query

    Returns:
        dist_query_cf: Similarity (L1) loss between counterfactual and query
    """
    cf_flat = torch.reshape(cf, (cf.shape[0], cf.shape[1] * cf.shape[2]))
    if X_query == []:
        X_flat = torch.zeros_like(cf_flat)
    else:
        X_flat = torch.reshape(X_query, (X_query.shape[0], X_query.shape[1] * X_query.shape[2]))
    dist_query_cf = torch.linalg.norm((cf_flat-X_flat), ord=1, dim=1)
    dist_query_cf /= cf.shape[1] * cf.shape[2] 
    return dist_query_cf

def compute_sparsity_loss(cf, X_query=[], freeze_indices=[]):
    """Penalizes high numbers of modified time steps and features as the L0 norm between query and counterfactual.

    Args:
        cf: counterfactual
        X_query: query
        freeze_indices: indices of immutable features

    Returns:
        dist_query_cf: Sparsity (L0) loss between counterfactual and query
    """
    cf_flat = torch.reshape(cf, (cf.shape[0], cf.shape[1] * cf.shape[2]))
    if X_query == []:
        X_flat = torch.zeros_like(cf_flat)
    else:
        X_flat = torch.reshape(X_query, (X_query.shape[0], X_query.shape[1] * X_query.shape[2]))
    dist_query_cf = torch.linalg.norm((cf_flat-X_flat), ord=0, dim=1)
    dist_query_cf /= cf.shape[1] * (cf.shape[2] - len(freeze_indices)) 
    return dist_query_cf

def compute_jerk_loss(deltas, device):
    """Penalizes large differences between modifications in consecutive time steps. 

    Args:
        deltas: deltas between query and counterfactual
        device: CPU or GPU

    Returns:
        jerk_loss: Jerk loss
    """
    deltas_extended = torch.zeros((deltas.shape[0], 1, deltas.shape[2])).to(device)
    deltas_extended = torch.cat((deltas_extended, deltas), dim=1)
    deltas_extended = deltas_extended[:,:-1,:]
    jerk_loss = torch.linalg.norm(deltas - deltas_extended, dim=2).sum(dim=1)
    jerk_loss /= deltas.shape[1] * deltas.shape[2] # normalize loss to be in range of other loss terms
    return jerk_loss