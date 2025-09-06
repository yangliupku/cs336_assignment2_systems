import torch
from jaxtyping import Float
import math
from einops import einsum
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_queries,
    N_keys,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_keys, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_keys, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_queries, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_queries,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Tk = tl.cdiv(N_keys, K_TILE_SIZE)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        prev_mi = mi
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        mi = tl.maximum(prev_mi, tl.max(Sij, axis=-1))
        Pij = tl.exp(Sij - mi[:, None])
        scaling_factor = tl.exp(prev_mi - mi)
        li = scaling_factor * li + tl.sum(Pij, axis=-1)
        Oi = scaling_factor[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    Oi = (1 / li)[:, None] * Oi
    li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, li, boundary_check=(0,))


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


class FlashAttentionV2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "B queries d_k"],
        K: Float[Tensor, "B keys d_k"],
        V: Float[Tensor, "B keys d_v"],
        is_causal: bool = False,
    ):
        ctx.save_for_backward(Q, K, V)
        Bq = 16
        Bk = 16
        D = Q.shape[2]
        B = Q.shape[0]
        N_queries = Q.shape[1]
        N_keys = K.shape[1]
        Tq = triton.cdiv(N_queries, Bq)
        O = torch.empty((B, N_queries, D), device=Q.device)
        L = torch.empty((B, N_queries), device=Q.device)
        scale = 1 / math.sqrt(D)
        grid = (Tq, B)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_queries,
            N_keys,
            scale,
            D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
        )
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
