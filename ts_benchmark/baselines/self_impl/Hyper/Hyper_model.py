# hyperbolic_timeseries.py
import torch
import torch.nn as nn
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds.stereographic import PoincareBall
from geoopt.manifolds.stereographic.math import mobius_matvec, mobius_add


# --- 小工具：安全读取 configs 字段 ---
def getcfg(cfg, name, default=None):
    return getattr(cfg, name, cfg.get(name, default) if isinstance(cfg, dict) else default)


def exp0(ball: PoincareBall, x): return ball.expmap0(x)
def log0(ball: PoincareBall, x): return ball.logmap0(x)
def proj(ball: PoincareBall, x): return ball.projx(x)


class MobiusLinear(nn.Module):
    """超曲率线性层：y = W ⊗ x ⊕ b（⊗=Möbius matvec, ⊕=Möbius add）"""
    def __init__(self, in_features, out_features, ball: PoincareBall):
        super().__init__()
        self.ball = ball
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # bias 放在流形上
        self.bias = ManifoldParameter(self.ball.random_normal(out_features), manifold=self.ball)

    def forward(self, xH):
        y = mobius_matvec(self.weight, xH, k=self.ball.c)
        y = mobius_add(y, self.bias, k=self.ball.c)
        return proj(self.ball, y)


class HypAct(nn.Module):
    """在切空间做激活：logmap0 -> act -> expmap0"""
    def __init__(self, ball: PoincareBall, act="gelu"):
        super().__init__()
        self.ball = ball
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()

    def forward(self, xH):
        xt = log0(self.ball, xH)
        yt = self.act(xt)
        yH = exp0(self.ball, yt)
        return proj(self.ball, yH)


class HypConv1d(nn.Module):
    """超曲率 1D 卷积：切空间做 Conv1d -> exp 回流形"""
    def __init__(self, channels, kernel_size=3, ball: PoincareBall=None):
        super().__init__()
        assert ball is not None
        self.ball = ball
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)

    def forward(self, xH):
        # xH: [B, L, C]
        xt = log0(self.ball, xH)      # [B, L, C]
        xt = xt.transpose(1, 2)       # -> [B, C, L]
        yt = self.conv(xt)
        yt = yt.transpose(1, 2)       # -> [B, L, C]
        yH = exp0(self.ball, yt)
        return proj(self.ball, yH)


class HyperModel(nn.Module):
    """
    完全超曲率的时序编码器（最简 CNN 版，参数来自 configs.xxx）
    输入 x: [B, L, enc_in] 欧式 -> exp0 -> 若干 (MobiusLinear + HypAct + HypConv1d)
    输出：若 head_euclidean=True，则回欧式线性头 [B, L, c_out]；否则输出超曲率表示
    """
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.configs = configs

        # ---- 从 configs.xxx 读取超参（带兜底）----
        d_in      = getcfg(configs, "enc_in", getcfg(configs, "d_in", 8))
        d_hidden  = getcfg(configs, "d_model", 64)
        d_out     = getcfg(configs, "c_out", getcfg(configs, "d_out", 8))
        layers    = getcfg(configs, "e_layers", getcfg(configs, "layers", 3))
        c         = getcfg(configs, "hyper_c", 1.0)   # 曲率；你也可以叫 c / curvature
        dropout   = getcfg(configs, "dropout", 0.0)
        act_name  = getcfg(configs, "activation", "gelu")
        self.head_euclidean = bool(getcfg(configs, "head_euclidean", True))

        # 保存关键维度（如需在外部引用）
        self.d_in, self.d_hidden, self.d_out = d_in, d_hidden, d_out
        self.layers_num = layers

        # 欧 -> 超（先做线性，再 exp0）
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.ball = PoincareBall(c=c)

        blocks = []
        for _ in range(layers):
            blocks += [
                MobiusLinear(d_hidden, d_hidden, self.ball),
                HypAct(self.ball, act=act_name),
                HypConv1d(d_hidden, kernel_size=3, ball=self.ball),
                nn.Dropout(dropout),
            ]
        self.blocks = nn.Sequential(*blocks)

        # 头部：欧式线性 / Möbius 线性
        if self.head_euclidean:
            self.out_head = nn.Linear(d_hidden, d_out)
        else:
            self.out_head = MobiusLinear(d_hidden, d_out, self.ball)

    def forward(self, x):
        # x: [B, L, enc_in] （欧式）
        h = self.in_proj(x)           # 欧式                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        hH = exp0(self.ball, h)       # -> 超曲率
        hH = proj(self.ball, hH)

        hH = self.blocks(hH)          # 超曲率块

        if self.head_euclidean:
            ht = log0(self.ball, hH)  # 切空间
            y = self.out_head(ht)     # 欧式输出 [B, L, c_out]
            return y
        else:
            yH = self.out_head(hH)    # 超曲率输出
            return yH


# --------- 自测（读取 configs.xxx）---------
if __name__ == "__main__":
    class Cfg(dict):
        """允许点取（cfg.xxx）也可当字典用（cfg['xxx']）"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    # 你的字典（示例，和你之前 DEFAULT_* 的键名一致即可）
    cfg = Cfg(
        enc_in=8, c_out=8, d_model=64, e_layers=2,
        dropout=0.1, activation="gelu",
        hyper_c=1.0, head_euclidean=True,
        device="cuda:0"
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    B, L = 16, 192
    x = torch.randn(B, L, cfg.enc_in, device=device)
    y = torch.randn(B, L, cfg.c_out, device=device)

    model = HyperModel(cfg).to(device)

    # Riemannian Adam：会识别 ManifoldParameter 并保持在流形上
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=3e-3)

    model.train()
    optimizer.zero_grad()
    pred = model(x)                         # 欧式头：直接 MSE
    print("x:", tuple(x.shape), "pred:", tuple(pred.shape))
    loss = ((pred - y) ** 2).mean()
    print("loss:", float(loss))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
