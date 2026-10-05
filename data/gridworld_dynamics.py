import torch

# 词表布局
X_OFFSET = 0      # 0-99    x 坐标
Y_OFFSET = 100    # 100-199 y 坐标
A_OFFSET = 200    # 200-203 动作
VOCAB_SIZE = 204
GRID_SIZE = 100
N_ACTIONS = 4

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def make_world(seed, grid_size=GRID_SIZE):
    """生成固定世界：每格类型。0 正常, 1 镜像, 2 传送。"""
    g = torch.Generator().manual_seed(seed)
    rand = torch.rand(grid_size, grid_size, generator=g)
    world = torch.zeros(grid_size, grid_size, dtype=torch.long)
    world[rand >= 0.90] = 1
    world[rand >= 0.97] = 2
    return world


def step(world, x, y, a):
    """根据 (x,y) 处的格子类型和动作 a，返回下一位置。"""
    t = world[x, y].item()
    dx, dy = ACTIONS[a]

    if t == 0:
        nx, ny = x + dx, y + dy
    elif t == 1:
        nx, ny = x - dx, y - dy
    else:
        nx = (x * 7 + 13) % GRID_SIZE
        ny = (y * 11 + 17) % GRID_SIZE

    nx = max(0, min(GRID_SIZE - 1, nx))
    ny = max(0, min(GRID_SIZE - 1, ny))
    return nx, ny


def make_sequence(world, length, batch_size, device, start_xy=None):
    """
    生成轨迹。
    返回:
        seq:    [B, L*3]  [x_1, y_1, a_1, x_2, y_2, a_2, ...]
        target: [B, L*3]  只在 (x_{t+1}, y_{t+1}) 位置有值，其余 -100
    """
    B = batch_size
    xs = torch.zeros(B, length, dtype=torch.long)
    ys = torch.zeros(B, length, dtype=torch.long)
    as_ = torch.zeros(B, length, dtype=torch.long)

    for b in range(B):
        if start_xy is None:
            x = torch.randint(0, GRID_SIZE, (1,)).item()
            y = torch.randint(0, GRID_SIZE, (1,)).item()
        else:
            x, y = start_xy

        for t in range(length):
            xs[b, t] = x
            ys[b, t] = y
            a = torch.randint(0, N_ACTIONS, (1,)).item()
            as_[b, t] = a
            x, y = step(world, x, y, a)

    seq = torch.stack([xs, ys, as_], dim=-1).reshape(B, length * 3)
    seq[:, 0::3] += X_OFFSET
    seq[:, 1::3] += Y_OFFSET
    seq[:, 2::3] += A_OFFSET

    target = torch.full_like(seq, -100)
    target[:, 3::3] = seq[:, 3::3]
    target[:, 4::3] = seq[:, 4::3]

    return seq.to(device), target.to(device)