import torch
import torch.nn.functional as F

def masked_bce_with_logits(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-6)

def train_overfit(model, loader, device="cpu", lr=1e-3, epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(epochs):
        total = 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = masked_bce_with_logits(logits, yb, mb)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {ep+1:02d} loss {total/len(loader):.4f}")
