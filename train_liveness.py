"""Huan luyen CNN texture. Du lieu: dataset/real/*.jpg va dataset/spoof/*.jpg
   python train_liveness.py --epochs 20
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config
from liveness import TextureCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs phai lon hon 0")
    if args.batch < 1:
        ap.error("--batch phai lon hon 0")
    if args.lr <= 0:
        ap.error("--lr phai lon hon 0")

    tf_train = transforms.Compose([
        transforms.Resize((config.CNN_INPUT, config.CNN_INPUT)),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomRotation(8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([transforms.Resize((config.CNN_INPUT, config.CNN_INPUT)),
                                 transforms.ToTensor()])

    full = datasets.ImageFolder(config.LIVENESS_DATASET)      # real=0, spoof=1
    print("Lop:", full.class_to_idx, "| tong:", len(full))
    if len(full) < 2:
        raise SystemExit("Dataset qua nho: can it nhat 1 anh cho moi lop real/spoof.")
    class_counts = {label: 0 for label in full.class_to_idx}
    for _, label in full:
        class_counts[next(name for name, idx in full.class_to_idx.items() if idx == label)] += 1
    if any(count < 2 for count in class_counts.values()):
        raise SystemExit(f"Moi lop can it nhat 2 anh. Phan bo hien tai: {class_counts}")
    n_val = max(1, int(0.2 * len(full)))
    train_idx, val_idx = random_split(range(len(full)), [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(42))

    class Sub(torch.utils.data.Dataset):
        def __init__(self, idx, tf):
            self.idx, self.tf = list(idx), tf
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            img, y = full[self.idx[i]]
            return self.tf(img), torch.tensor([float(y)])

    train_dl = DataLoader(Sub(train_idx, tf_train), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(Sub(val_idx, tf_val), batch_size=args.batch)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextureCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total
        print(f"Epoch {ep:02d} | val acc = {acc:.3f}")
        if acc >= best:
            best = acc
            config.LIVENESS_MODEL.parent.mkdir(exist_ok=True)
            torch.save(model.cpu().state_dict(), config.LIVENESS_MODEL)
            model.to(dev)
    print(f"Xong. Best val acc = {best:.3f} -> {config.LIVENESS_MODEL}")


if __name__ == "__main__":
    main()
