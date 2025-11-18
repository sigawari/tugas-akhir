import os, random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AmpScaler:
    """Wrapper aman AMP: on kalau CUDA tersedia, off kalau CPU."""
    def __init__(self, enabled=None):
        self.enabled = torch.cuda.is_available() if enabled is None else enabled
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.enabled)
        self.autocast = torch.cuda.amp.autocast if self.enabled else _DummyAutocast()

    def scale(self, loss):  return self.scaler.scale(loss) if self.enabled else loss
    def step(self, opt):    self.scaler.step(opt) if self.enabled else opt.step()
    def update(self):       self.scaler.update() if self.enabled else None

class _DummyAutocast:
    def __call__(self): return self
    def __enter__(self): pass
    def __exit__(self, *args): pass

@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def save_best(model, acc, fold, outdir="checkpoints", tag="best"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"resnet2d_fold{fold}_{tag}.pth")
    torch.save({"model": model.state_dict(), "acc": acc}, path)
    return path
