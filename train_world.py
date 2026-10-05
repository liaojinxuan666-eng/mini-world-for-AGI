import sys
import time
import torch
import torch.nn.functional as F

from model.world_model import WorldModel
from data.gridworld_dynamics import (
    make_world, make_sequence, VOCAB_SIZE,
)


def eval_single(model, world, length, device):
    model.eval()
    with torch.no_grad():
        seq, target = make_sequence(world, length, batch_size=128, device=device)
        logits = model(seq)
        pred = logits[:, :-1].argmax(-1)             # [B, L-1]
        mask = target[:, 1:] != -100                  # [B, L-1]
        return (pred[mask] == target[:, 1:][mask]).float().mean().item()


def eval_rollout(model, world, prefix_len=8, rollout_len=50,
                 batch_size=64, max_ctx=32, device="cuda"):
    """
    从真实前缀出发，自回归预测 rollout_len 步。
    返回每个 rollout 步的 acc 列表。
    """
    model.eval()
    total_len = prefix_len + rollout_len
    seq, _ = make_sequence(world, length=total_len,
                            batch_size=batch_size, device=device)

    cur = seq[:, :prefix_len * 3].clone()
    correct = torch.zeros(rollout_len, device=device)

    with torch.no_grad():
        for t in range(rollout_len):
            gt_idx = (prefix_len + t) * 3

            # 预测 x
            logits = model(cur)
            pred_x = logits[:, -1, :].argmax(-1)
            cur = torch.cat([cur, pred_x.unsqueeze(1)], dim=1)

            # 预测 y
            logits = model(cur)
            pred_y = logits[:, -1, :].argmax(-1)
            cur = torch.cat([cur, pred_y.unsqueeze(1)], dim=1)

            # 记录对错
            gt_x = seq[:, gt_idx]
            gt_y = seq[:, gt_idx + 1]
            correct[t] = ((pred_x == gt_x) & (pred_y == gt_y)).float().sum()

            # 拼真实动作 a，继续 rollout
            a_t = seq[:, gt_idx + 2]
            cur = torch.cat([cur, a_t.unsqueeze(1)], dim=1)

            # 截断上下文，避免越跑越慢
            if cur.shape[1] > max_ctx * 3:
                cur = cur[:, -max_ctx * 3:]

    return (correct / batch_size).cpu().tolist()


def train_one(length=32, batch_size=32, steps=3000,
              lr=3e-4, log_every=500, seed=42, device="cuda"):
    torch.manual_seed(seed)
    world = make_world(seed=seed)

    model = WorldModel(vocab_size=VOCAB_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        model.train()
        seq, target = make_sequence(world, length, batch_size, device)
        logits = model(seq)
        loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, VOCAB_SIZE),
    target[:, 1:].reshape(-1),
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

    torch.save(model.state_dict(), "/kaggle/working/world_model.pt")

    # 单步对照
    acc_same = eval_single(model, make_world(seed=seed), length, device)
    acc_new = eval_single(model, make_world(seed=999), length, device)
    print(f"\nparams={n_params:,}  time={time.time()-t0:.0f}s")
    print(f"单步 acc（训练世界）: {acc_same:.4f}")
    print(f"单步 acc（全新世界）: {acc_new:.4f}")

    # 关键：rollout
    print("\n=== Rollout 评估 ===")
    for name, w in [("训练世界", make_world(seed=seed)),
                    ("全新世界", make_world(seed=999))]:
        accs = eval_rollout(model, w, prefix_len=8, rollout_len=50,
                            batch_size=64, device=device)
        # 打印关键点
        points = [1, 5, 10, 20, 30, 50]
        print(f"\n{name}:")
        for p in points:
            if p <= len(accs):
                print(f"  acc@{p:<3d} = {accs[p-1]:.4f}")


if __name__ == "__main__":
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    train_one(length=length, steps=steps)