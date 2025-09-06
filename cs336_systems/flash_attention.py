import torch
from jaxtyping import Float
import math
from einops import einsum
from torch import Tensor


class FlashAttentionV2Torch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... keys d_v"],
        is_causal: bool = False,
    ):
        ctx.save_for_backward(Q, K, V)
        Bq = 16
        Bk = 16
        Q_input_shape = Q.shape
        K_input_shape = K.shape
        V_input_shape = V.shape
        D = Q.shape[-1]
        Q = Q.view(-1, Q.shape[-2], Q.shape[-1])
        K = K.view(-1, K.shape[-2], K.shape[-1])
        V = V.view(-1, V.shape[-2], V.shape[-1])
        B = Q.shape[0]
        Tq = math.ceil(Q.shape[-2] / Bq)
        Tk = math.ceil(K.shape[-2] / Bk)
        OO = []
        LL = []
        for i in range(Tq):
            Oi = torch.zeros(B, Bq, V.shape[-1])
            li = torch.zeros(
                B,
                Bq,
            )
            mi = torch.ones(
                B,
                Bq,
            ) * torch.tensor(float("-inf"))
            for j in range(Tk):
                Qi = Q[:, i * Bq : (i + 1) * Bq, :]
                Kj = K[:, j * Bk : (j + 1) * Bk, :]
                Vj = V[:, j * Bk : (j + 1) * Bk, :]
                prev_mi = mi
                Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") / math.sqrt(D)
                mi = torch.maximum(prev_mi, torch.max(Sij, axis=-1).values)
                Pij = torch.exp(Sij - mi.unsqueeze(-1))
                scaling_factor = torch.exp(prev_mi - mi)
                li = scaling_factor * li + torch.sum(Pij, axis=-1)
                Oi = scaling_factor.unsqueeze(-1) * Oi + Pij @ Vj

            Oi = (1 / li).unsqueeze(-1) * Oi
            li = mi + torch.log(li)
            OO.append(Oi)
            LL.append(li)
        O = torch.concat(OO, axis=1).view(*Q_input_shape[:-1], V_input_shape[-1])
        L = torch.concat(LL, axis=1).view(*Q_input_shape[:-1])
        ctx.save_for_backward(L, O)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


if __name__ == "__main__":
    B = 8
    S = 64
    D = 8
    Q = torch.rand(B, S, D)
    K = torch.rand(B, S, D)
    V = torch.rand(B, S, D)
    S = einsum(Q, K, "... bq d, ... bk d -> ... bq bk") / math.sqrt(D)
    P = torch.softmax(S, dim=-1)
    O = einsum(P, V, "... bq bk, ... bk d ->... bq d")
    L = torch.log(torch.sum(torch.exp(S), axis=-1))
    print("O", O[0, 20])
    print(O.shape)
    # print("L", L)
    # print(L.shape)
    O = FlashAttentionV2Torch.apply(Q, K, V)
    print("O", O[0, 20])
    print(O.shape)
