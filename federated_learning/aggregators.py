import torch
from copy import deepcopy

def _flatten_tensors(d):
    """
    Flattens a dictionary of tensors into a single tensor,
    ensuring consistent data type and dimensionality.
    """
    parts = []
    shapes = {}
    keys = []
    
    for k, v in d.items():
        # Move the tensor to CPU and convert to Float for consistency
        v_cpu = v.cpu().to(torch.float32)
        if v_cpu.dim() == 0:
            v_cpu = v_cpu.unsqueeze(0)
        # Ensure the tensor is 1D, handling scalars (0-D tensors)
        parts.append(v_cpu.flatten())
        
        shapes[k] = v.shape
        keys.append(k)
    
    # Concatenate all parts into a single tensor.
    # .flatten() and .to(torch.float32) guarantee compatibility.
    flat = torch.cat(parts)
    
    return flat, shapes, keys

def _unflatten_to_state(flat, shapes, keys, device=None):
    """
    Restores a flattened tensor to a state dictionary.

    Args:
        flat (torch.Tensor): The flattened 1D tensor of model parameters.
        shapes (dict): A dictionary mapping parameter keys to their original shapes.
        keys (list): A list of parameter keys in the order they were flattened.
        device (torch.device, optional): The device to move the resulting tensors to. Defaults to None.

    Returns:
        dict: The restored state dictionary with tensors of their original shapes.
    """
    state = {}
    idx = 0
    # The fix is here: iterate over the keys and access the shapes dictionary
    for k in keys:
        s = shapes[k]  # Get the shape for the current key
        n = 1
        for dim in s:
            n *= dim
        piece = flat[idx:idx+n]
        state[k] = piece.view(s).to(device)
        idx += n
    return state

def some_mean_filter_then_average(deltas, z_threshold=2.0):
    """
    Aggregates client model deltas using a defense mechanism.

    Args:
        deltas (list): A list of state_dicts (each value is a tensor) from clients.
        z_threshold (float, optional): The z-score threshold for filtering. Defaults to 2.0.

    Returns:
        tuple: A tuple containing the aggregated state dictionary and an info dictionary.
    """
    n = len(deltas)
    if n == 0:
        raise ValueError("no deltas provided")
    if n == 1:
        return deltas[0], {"kept_indices": [0], "filtered_count": 0, "distances": [0.0]}

    # flatten all deltas
    flats = []
    shapes = None
    keys = None
    for d in deltas:
        flat, shapes, keys = _flatten_tensors(d)
        flats.append(flat)
    stacked = torch.stack(flats, dim=0)  # (n, dim)

    # compute mean delta vector
    mean_vec = torch.mean(stacked, dim=0)

    # distances to mean
    diffs = stacked - mean_vec.unsqueeze(0)
    dists = torch.norm(diffs, dim=1)  # (n,)

    mu = dists.mean()
    sigma = dists.std(unbiased=False)

    threshold = mu + z_threshold * (sigma + 1e-12)
    mask = (dists <= threshold)
    kept_idx = mask.nonzero(as_tuple=False).view(-1).tolist()

    # fallback: keep half closest to mean if all filtered
    if len(kept_idx) == 0:
        k = max(1, n // 2)
        _, idxs = torch.topk(-dists, k)  # smallest distances
        kept_idx = idxs.tolist()

    kept = stacked[kept_idx]
    mean_flat_kept = torch.mean(kept, dim=0)
    agg_state = _unflatten_to_state(mean_flat_kept, shapes, keys)

    info = {
        "kept_indices": kept_idx,
        "filtered_count": n - len(kept_idx),
        "distances": [float(x) for x in dists.tolist()],
        "mu_distance": float(mu.item()),
        "sigma_distance": float(sigma.item()),
        "threshold": float(threshold.item())
    }
    return agg_state, info


# Median strategy 

def median_filter_then_average(deltas):
    """
    Aggregates deltas by taking the coordinate-wise median.
    """
    if len(deltas) == 0:
        raise ValueError("No deltas provided")
    if len(deltas) == 1:
        return deltas[0], {"kept_indices": [0], "filtered_count": 0}

    flats = []
    shapes = None
    keys = None
    for d in deltas:
        flat, shapes, keys = _flatten_tensors(d)
        flats.append(flat)
    stacked = torch.stack(flats, dim=0)

    # Compute the median along the client dimension (dim=0)
    median_delta_flat = torch.median(stacked, dim=0)[0]

    # Unflatten the median delta to restore original model structure
    agg_state = _unflatten_to_state(median_delta_flat, shapes, keys)
    
    info = {
        "filtered_count": 0,
        "kept_indices": list(range(len(deltas))),
    }
    return agg_state, info

# Krum 

def krum_filter_then_average(deltas, f, m=1):
    """
    Performs Krum aggregation to select the most trustworthy update.

    Args:
        deltas (list): A list of state_dicts (each a client's delta).
        f (int): The number of Byzantine (malicious) clients.
        m (int): The number of clients to select (Multi-Krum).

    Returns:
        tuple: The aggregated delta state_dict and an info dictionary.
    """
    n = len(deltas)
    if n <= 2 * f + m:
        raise ValueError("Number of clients is too small for Krum.")
    
    flats = []
    shapes = None
    keys = None
    for d in deltas:
        flat, shapes, keys = _flatten_tensors(d)
        flats.append(flat)
    stacked = torch.stack(flats, dim=0)

    # Compute pairwise Euclidean distances
    pairwise_dists = torch.cdist(stacked, stacked, p=2)

    scores = []
    # For each client, compute the sum of distances to its n-f-2 closest neighbors
    for i in range(n):
        # Sort distances to find the smallest n-f-2
        sorted_dists = torch.sort(pairwise_dists[i])[0]
        # Sum the distances to the n-f-2 closest clients
        score = torch.sum(sorted_dists[:n - f - 2])
        scores.append(score)

    # Select the client(s) with the lowest Krum score
    scores = torch.tensor(scores)
    _, top_m_indices = torch.topk(scores, m, largest=False)
    
    kept_idx = top_m_indices.tolist()

    if m == 1:
        agg_flat = stacked[kept_idx[0]]
    else: # Multi-Krum
        agg_flat = torch.mean(stacked[kept_idx], dim=0)
    
    agg_state = _unflatten_to_state(agg_flat, shapes, keys)
    
    info = {
        "kept_indices": kept_idx,
        "filtered_count": n - len(kept_idx),
    }
    return agg_state, info

# Trimmed mean 

def trimmed_mean_filter_then_average(deltas, b):
    """
    Aggregates deltas using the trimmed mean.

    Args:
        deltas (list): A list of state_dicts (each a client's delta).
        b (int): The number of updates to trim from each end.

    Returns:
        tuple: The aggregated delta state_dict and an info dictionary.
    """
    n = len(deltas)
    if n <= 2 * b:
        raise ValueError("Number of clients is too small for trimmed mean.")
    
    flats = []
    shapes = None
    keys = None
    for d in deltas:
        flat, shapes, keys = _flatten_tensors(d)
        flats.append(flat)
    stacked = torch.stack(flats, dim=0)
    
    # Sort the stacked tensor along the client dimension (dim=0)
    # This sorts each parameter's values independently
    sorted_stacked, _ = torch.sort(stacked, dim=0)
    
    # Trim b values from the top and b from the bottom
    trimmed_stacked = sorted_stacked[b:n-b]
    
    # Compute the mean of the trimmed values
    agg_flat = torch.mean(trimmed_stacked, dim=0)
    
    agg_state = _unflatten_to_state(agg_flat, shapes, keys)
    
    info = {
        "filtered_count": 2 * b,
        "kept_indices": "Not applicable",
    }
    return agg_state, info