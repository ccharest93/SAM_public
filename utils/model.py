import torch
import time
from pytorch_memlab import MemReporter

def to_device(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    else:
        return obj

class GPUManager():
    def __init__(self, tensors,task, device=torch.device('cuda'), verbose=False):
        self.tensors = tensors
        self.device = device
        self.task = task
        self.total_time = 0
        self.reporter = MemReporter()
        self.verbose = verbose
    
    def __enter__(self):
        start = time.perf_counter()
        self.tensors = to_device(self.tensors, self.device)
        if self.verbose:
            print("ENTERING...")
            print("Context: {}".format(self.task))
            print("Device: {}".format(self.device)) 
            self.reporter.report()
            self.total_time += time.perf_counter() - start
        return self.tensors
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.verbose:
            print("EXITING...")
            print("Context: {}".format(self.task))
            print("Time taken: {}".format(self.total_time))
            self.reporter.report()
