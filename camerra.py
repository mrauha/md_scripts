import numpy as np
from scipy.spatial.distance import pdist
def get_dynamic_residue_pairs(X, step=10, cutoff=10.0, jmin=0.05, jmax=0.95):
    """
    Calculates dynamic residue pairs using the CAMERRA method 
    
    https://doi.org/10.1002/jcc.25192
    
    Calculates distance matrix for each frame, applies a binary mask
    that maps contacts under cutoff to 1, other to 0. These are then
    averaged, and those pairs with average contact between jmin and 
    jmax are collected. Idea is to filter out the residues that are 
    almost all the time in/not in contact, ie the dynamic residues pairs
    
    Parameters:
    
    X: np.array (n_atoms,3,n_frames), calpha coordinates
    step: int, consider only every n'th frame
    cutoff: float, distance which determines if residues are in contact
    jmin, jmax: float between 0-1. the residues that have average contact value
                between jmin and jmax are kept. 0.05,0.95 are typically used.
    
    Returns: 
    pairs: list of list containing pair of ints, the indices of dynamic residue pairs
    
    """
    n = X.shape[0]
    n_upper_diagonal = int(n * (n - 1) / 2)
    us = np.zeros(n_upper_diagonal)
    for frame in range(0, X.shape[-1], step):
        # unique elements/upper diagonal of distance matrix in a vector form
        u = pdist(X[:, :, frame])
        # apply mask
        u[u < cutoff] = 1
        u[u >= cutoff] = 0
        us += u
    #average
    us = us / X.shape[-1]  
    dynamic_idx = np.argwhere((us >= jmin) & (us <= jmax))
    # convert get the indices of pairs
    pairs = [
        [i[0], j[0]]
        for i, j in zip(*np.unravel_index(dynamic_idx, shape=(n, n), order="C"))
    ]
    print(f"{100*len(pairs)/n_upper_diagonal:.1f}% of contacts are dynamic! :)")
    return pairs