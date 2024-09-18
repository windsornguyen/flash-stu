import gc

class GarbageCollection:
    def __init__(self, gc_freq=1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        gc.collect(1)

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            gc.collect(1)

def linear_decay_with_warmup( # https://arxiv.org/pdf/2310.07831
    current_step: int, 
    warmup_steps: int, 
    num_steps: int, 
    max_lr: float = 3e-4, 
    min_lr: float = 3e-5,
) -> float:
    if current_step < warmup_steps:
        return min_lr + (max_lr - min_lr) * float(current_step) / float(max(warmup_steps, 1))
    else:
        return max_lr - (max_lr - min_lr) * float(current_step - warmup_steps) / float(max(num_steps - warmup_steps, 1))
