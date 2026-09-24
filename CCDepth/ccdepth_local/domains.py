import numpy as np
from scipy.ndimage import binary_erosion, label, find_objects
from .contracts import DomainData

def build_domains(inputs):
    o=inputs.mask_valid; v=inputs.dem_valid; support=o&inputs.flood&v
    if not support.any(): raise ValueError('Empty support')
    ids,n=label(support,structure=np.ones((3,3),bool))
    hard=binary_erosion(support,structure=np.ones((3,3),bool),border_value=0)
    counts=np.bincount(ids.ravel());boxes=find_objects(ids)
    table=[dict(id=i,pixels=int(counts[i]),bbox=boxes[i-1]) for i in range(1,n+1)]
    return DomainData(o,support,o&~inputs.flood&v,hard,support&~hard,support&~hard,ids,table)
