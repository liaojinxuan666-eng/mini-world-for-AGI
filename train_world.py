import sys
import time
import torch
import torch.nn.functional as F

from model.world_model import WorldModel
from data.gridworld_dynamics import (
    make_world, make_sequence, VOCAB_SIZE,
)


def make_batch(world, length, batch_size, device):
    return make_sequence(world, length, batch_size, device)


def train_one(
    length=32,
    batch_size=32,
    steps=3000,
    lr=3e-4,
    log_every=100,
    seed=42,
    device="cuda",
):
    torch.manual_seed(seed)
    world = make_world(seed=seed)

    model = WorldModel(vocab_size=VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        seq, target = make_batch(world, length, batch_size, device)
        logits = model(seq)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target.reshape(-1),
            ignore_index=-100,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {step}/{steps}  loss {avg:.4f}  elapsed {time.time()-t0:.0f}s")

    # 评估：单步准确率
    model.eval()
    with torch.no_grad():
        seq, target = make_batch(world, length, 128, device)
        logits = model(seq)
        pred = logits.argmax(-1)                    # [B, L]
        mask = target != -100                       # [B, L]
        acc = (pred[mask] == target[mask]).float().mean().item()

    print(f"params={n_params:,}  single_step_acc={acc:.4f}  time={time.time()-t0:.0f}s")
    return model, world, acc


if __name__ == "__main__":
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    train_one(length=length, steps=steps)