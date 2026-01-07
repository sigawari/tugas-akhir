import wandb
import inspect

print("wandb module:", wandb)
print("wandb file  :", getattr(wandb, "__file__", None))
print("has init?   :", hasattr(wandb, "init"))
