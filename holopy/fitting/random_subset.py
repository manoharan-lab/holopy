import numpy as np
from holopy.core import Marray

def make_subset_data(data, random_subset):
    if random_subset is not None:
        n_sel = int(np.ceil(data.size*random_subset))
        selection = np.random.choice(data.size, n_sel, replace=False)
        subset = data.ravel()[selection]
        positions = data.positions.xyz()[selection]
        return Marray(subset, positions=positions,
                             origin=data.origin,
                             optics=data.optics)
