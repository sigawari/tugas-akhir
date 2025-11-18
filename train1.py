# train.py
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets.bisindo_dataset import BISINDOResNetDatasetKFold, collate_fn
from models.resnet2d_landmark import ResNet2DForLandmarks
from utils.runtime import set_seed, get_device, AmpScaler, accuracy, save_best

def train_one_epoch(model, loader, device, opt, crit, amp):
    model.train(); tot_loss=0; tot_correct=0; tot_total=0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = model(X)
            loss   = crit(logits, y)
        amp.scale(loss).backward()
        amp.step(opt); amp.update()
        tot_loss   += loss.item()*y.size(0)
        tot_correct+= (logits.argmax(1)==y).sum().item()
        tot_total  += y.numel()
    return tot_loss/tot_total, tot_correct/tot_total

@torch.no_grad()
def eval_epoch(model, loader, device, crit):
    model.eval(); tot_loss=0; tot_correct=0; tot_total=0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = crit(logits, y)
        tot_loss   += loss.item()*y.size(0)
        tot_correct+= (logits.argmax(1)==y).sum().item()
        tot_total  += y.numel()
    return tot_loss/tot_total, tot_correct/tot_total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_classes", type=int, default=2)
    ap.add_argument("--face_stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)  # Windows aman = 0
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")  # paksa non-AMP
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    amp = AmpScaler(enabled=(not args.no_amp))
    print(f"Device: {device} | AMP: {amp.enabled}")

    modal_cfg = {"use_pose":True, "use_hands":True, "use_face":True}

    train_ds = BISINDOResNetDatasetKFold(args.fold, True, args.split_dir, modal_cfg, args.face_stride, jitter_std=0.01)
    val_ds   = BISINDOResNetDatasetKFold(args.fold, False, args.split_dir, modal_cfg, args.face_stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)

    model = ResNet2DForLandmarks(in_ch=9, n_classes=args.n_classes).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit  = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt, crit, amp)
        va_loss, va_acc = eval_epoch(model, val_loader, device, crit)
        print(f"Epoch {ep:02d}/{args.epochs} | train: loss {tr_loss:.4f} acc {tr_acc:.3f} | val: loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            path = save_best(model, best, args.fold, outdir="checkpoints", tag="best")
            print(f"  ✅ Saved: {path} (best val acc={best:.3f})")

if __name__ == "__main__":
    main()
