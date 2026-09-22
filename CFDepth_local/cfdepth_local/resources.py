import ctypes
import os
import numpy as np

def memory_info():
    if os.name=='nt':
        class MEM(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','avail','page_total','page_avail','virtual_total','virtual_avail','extended')]
        m=MEM();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.total),int(m.avail)
    import psutil
    m=psutil.virtual_memory();return m.total,m.available

def peak_rss():
    if os.name=='nt':
        class PMC(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak','working','page_peak','page','nonpage_peak','nonpage','pagefile','pagefile_peak')]
        p=PMC();p.cb=ctypes.sizeof(p)
        handle=ctypes.windll.kernel32.GetCurrentProcess
        handle.restype=ctypes.c_void_p
        func=ctypes.windll.psapi.GetProcessMemoryInfo;func.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        func(handle(),ctypes.byref(p),p.cb);return int(p.peak)
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

class Allocator:
    def __init__(self,root,budget):self.root=root;self.root.mkdir(parents=True,exist_ok=True);self.budget=budget;self.used=0
    def __call__(self,name,shape,dtype):
        size=int(np.prod(shape))*np.dtype(dtype).itemsize
        self.used+=size
        if self.used>self.budget:
            return np.lib.format.open_memmap(self.root/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
        return np.empty(shape,dtype=dtype)
