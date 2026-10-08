"""Treat legacy all-None edge arrays as absent, without inventing features."""
import sys, importlib.util, numpy as np
path='/local-scratch2/mirzaei/fb/GraphVAE-REQ/scripts/evaluate_attributed_graph_realism_checkpoints.py'
spec=importlib.util.spec_from_file_location('proteins_attributed_eval',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=m._cache_split
def split(cache,name):
    a,n,e=original(cache,name)
    if e is not None and all(x is None or np.asarray(x).size==0 for x in e):
        e=None
    return a,n,e
m._cache_split=split
m.main()
