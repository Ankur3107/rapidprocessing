from fastprogress import *
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import functools, operator

def helper_fn(func):
    global inner_fn
    def inner_fn(x):
        i, x_i = x
        print('',end=' ')
        return i, func(x_i)
    return inner_fn

def parallel(func, items, chunksize=None, max_workers=1):
    if isinstance(items, list):
        if chunksize:
            arr = [items[i: i+chunksize] for i in (range(0, len(items), chunksize))]
        else:
            chunksize = int(len(items)/max_workers)
            arr = [items[i: i+chunksize] for i in (range(0, len(items), chunksize))]
            
    elif isinstance(items, np.ndarray):
        if chunksize:
            n_splits = int((items.shape[0]/chunksize)+0.5)
            arr = np.array_split(items, n_splits)
        else:
            arr = np.array_split(items, max_workers)
    
    main_fn = helper_fn(func)
    
    if max_workers<2: results = list(progress_bar(map(main_fn, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(main_fn, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results


def do_multiprocessing(f_py=None, chunksize=None, max_workers=1):
    assert callable(f_py) or f_py is None
    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            items = args[0]
            results = parallel(func, items, chunksize=chunksize, max_workers=max_workers)
            results = functools.reduce(operator.iconcat,[r[1] for r in results] , [])
            return results
        return wrapper
    return _decorator(f_py) if callable(f_py) else _decorator
