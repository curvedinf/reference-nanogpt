from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import triton
import triton.language as tl

NUM_SUBSPACES = 64
SUBSPACE_RANK = 8
TOPK_NEIGHBORS = 6
NUM_SPLANIFOLDS = 3
LOCAL_MLP_HIDDEN_DIM = 2574
GLOBAL_MLP_HIDDEN_DIM = 256
BASELINE_MLP_PARAMS_768 = 4718592
ONE_TENTH_TARGET_PARAMS_768 = BASELINE_MLP_PARAMS_768 / 10.0
ONE_TENTH_NUM_SUBSPACES = 18
ONE_TENTH_SUBSPACE_RANK = 8
ONE_TENTH_TOPK_NEIGHBORS = 6
ONE_TENTH_NUM_SPLANIFOLDS = 2
ONE_TENTH_LOCAL_MLP_HIDDEN_DIM = 256
ONE_TENTH_GLOBAL_MLP_HIDDEN_DIM = 93
ONE_TENTH_TOTAL_PARAMS_768 = 471867
WEIGHT_DTYPE = torch.bfloat16
SPLANIFOLD_SIGMA = 3.0
SPLANIFOLD_EXTRAPOLATION = 0.0

ROUTING_TEMP_SCALE = 2.0

if TOPK_NEIGHBORS >= NUM_SUBSPACES / 10:
    raise ValueError("invalid SSINF3 constants: expected topk_neighbors < num_subspaces / 10")
_regression_dim = TOPK_NEIGHBORS * SUBSPACE_RANK
if not (32 <= _regression_dim <= 64):
    raise ValueError("invalid SSINF3 constants: expected 32 <= topk_neighbors * subspace_rank <= 64")
_one_tenth_regression_dim = ONE_TENTH_TOPK_NEIGHBORS * ONE_TENTH_SUBSPACE_RANK
if _one_tenth_regression_dim <= 32:
    raise ValueError("invalid one-tenth SSINF3 constants: expected topk_neighbors * subspace_rank > 32")

_SSINF_SSM_FUSED_CONFIGS = [
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
]


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x) + 1e-6)


def _inv_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _compute_sigma(raw: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    sigma = F.softplus(raw) + sigma_min
    return sigma.clamp(max=sigma_max)


def _compute_extrapolation(raw: torch.Tensor, extrap_max: float) -> torch.Tensor:
    return torch.tanh(raw) * extrap_max


def _compute_dtype(weight_dtype: torch.dtype) -> torch.dtype:
    return weight_dtype


def _cubic_hermite(p0: Tensor, p1: Tensor, v0: Tensor, v1: Tensor, t: Tensor) -> Tensor:
    if t.ndim == 1:
        t = t.unsqueeze(1)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h01 * p1 + h10 * v0 + h11 * v1


def k_splanifold_eval(
    coords: Tensor,
    anchor_start: Tensor,
    anchor_end: Tensor,
    basis_start: Tensor,
    basis_end: Tensor,
    pos_tangent_start: Tensor,
    pos_tangent_end: Tensor,
    basis_tangent_start: Tensor,
    basis_tangent_end: Tensor,
    sigma: Tensor,
    extrapolation: Tensor,
) -> Tensor:
    rank = int(coords.shape[1])
    sigma = sigma.reshape(-1, 1)
    extrapolation = extrapolation.reshape(-1, 1)

    u = coords * (1 + 2 * extrapolation) - extrapolation
    t = u.mean(dim=1, keepdim=True)
    delta = u - t

    sum_u = u.sum(dim=1, keepdim=True)
    sum_abs = sum_u.abs()
    u_max = u.abs().max(dim=1, keepdim=True).values
    sum_eps = torch.clamp(u_max * 1e-3, min=1e-6)
    use_fallback = sum_abs < sum_eps
    safe_sum = torch.where(use_fallback, torch.where(sum_u >= 0, sum_eps, -sum_eps), sum_u)
    w = u / safe_sum
    w = torch.where(use_fallback.expand_as(w), torch.full_like(w, 1.0 / float(rank)), w)

    pos_start_delta = pos_tangent_start - anchor_start.unsqueeze(1)
    pos_end_delta = pos_tangent_end - anchor_end.unsqueeze(1)
    v_start = torch.bmm(w.unsqueeze(1), pos_start_delta).squeeze(1) * sigma
    v_end = torch.bmm(w.unsqueeze(1), pos_end_delta).squeeze(1) * sigma

    d_start = torch.bmm(delta.unsqueeze(1), basis_start).squeeze(1)
    d_end = torch.bmm(delta.unsqueeze(1), basis_end).squeeze(1)
    t_start = torch.bmm(delta.unsqueeze(1), basis_tangent_start - basis_start).squeeze(1) * sigma
    t_end = torch.bmm(delta.unsqueeze(1), basis_tangent_end - basis_end).squeeze(1) * sigma

    spine = _cubic_hermite(anchor_start, anchor_end, v_start, v_end, t)
    displacement = _cubic_hermite(d_start, d_end, t_start, t_end, t)
    return spine + displacement


def ssinf3_torch_reference_forward(
    input_batch: Tensor,
    input_basis_matrix: Tensor,
    center_projection: Tensor,
    output_basis: Tensor,
    splanifold_anchor_start: Tensor,
    splanifold_anchor_end: Tensor,
    splanifold_basis_start: Tensor,
    splanifold_basis_end: Tensor,
    splanifold_pos_tangent_start: Tensor,
    splanifold_pos_tangent_end: Tensor,
    splanifold_basis_tangent_start: Tensor,
    splanifold_basis_tangent_end: Tensor,
    splanifold_sigma: Tensor,
    splanifold_extrapolation: Tensor,
    local_mlp_weight_in: Tensor,
    local_mlp_bias_in: Tensor,
    local_mlp_weight_out: Tensor,
    local_mlp_bias_out: Tensor,
    local_mlp_weight_gate: Tensor,
    local_mlp_bias_gate: Tensor,
    global_mlp_weight_in: Tensor,
    global_mlp_bias_in: Tensor,
    global_mlp_weight_out: Tensor,
    global_mlp_bias_out: Tensor,
    topk_neighbors: int,
    routing_temperature: float,
    activation: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    if input_batch.ndim != 2:
        raise ValueError(f"input_batch must be 2D, got {tuple(input_batch.shape)}")

    batch_tokens, _ = input_batch.shape
    num_subspaces = int(center_projection.shape[0])
    subspace_rank = int(center_projection.shape[1])
    num_splanifolds = int(splanifold_anchor_start.shape[1])
    output_dim = int(output_basis.shape[2])
    local_hidden = int(local_mlp_weight_in.shape[1])
    if batch_tokens == 0:
        return input_batch.new_zeros((0, output_dim))

    topk_neighbors = int(topk_neighbors)
    if topk_neighbors <= 0:
        raise ValueError("topk_neighbors must be positive.")
    if topk_neighbors > num_subspaces:
        raise ValueError(f"topk_neighbors ({topk_neighbors}) must be <= num_subspaces ({num_subspaces}).")
    routing_temperature = float(routing_temperature)
    if routing_temperature <= 0:
        raise ValueError("routing_temperature must be > 0.")

    act_fn = activation or (lambda x: F.gelu(x, approximate="tanh"))
    compute_dtype = _compute_dtype(input_basis_matrix.dtype)
    device = input_batch.device

    x = torch.nan_to_num(input_batch.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    input_basis = torch.nan_to_num(input_basis_matrix.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    centers = torch.nan_to_num(center_projection.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    output_basis = torch.nan_to_num(output_basis.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    anchor_start = torch.nan_to_num(splanifold_anchor_start.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    anchor_end = torch.nan_to_num(splanifold_anchor_end.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    basis_start = torch.nan_to_num(splanifold_basis_start.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    basis_end = torch.nan_to_num(splanifold_basis_end.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    pos_tangent_start = torch.nan_to_num(splanifold_pos_tangent_start.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    pos_tangent_end = torch.nan_to_num(splanifold_pos_tangent_end.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    basis_tangent_start = torch.nan_to_num(splanifold_basis_tangent_start.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    basis_tangent_end = torch.nan_to_num(splanifold_basis_tangent_end.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    sigma = torch.nan_to_num(splanifold_sigma.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    extrapolation = torch.nan_to_num(splanifold_extrapolation.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    local_w_in = torch.nan_to_num(local_mlp_weight_in.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    local_b_in = torch.nan_to_num(local_mlp_bias_in.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    local_w_out = torch.nan_to_num(local_mlp_weight_out.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    local_b_out = torch.nan_to_num(local_mlp_bias_out.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    local_w_gate = torch.nan_to_num(local_mlp_weight_gate.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    local_b_gate = torch.nan_to_num(local_mlp_bias_gate.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    global_in_w = torch.nan_to_num(global_mlp_weight_in.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    global_in_b = torch.nan_to_num(global_mlp_bias_in.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    global_out_w = torch.nan_to_num(global_mlp_weight_out.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    global_out_b = torch.nan_to_num(global_mlp_bias_out.to(dtype=compute_dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0).contiguous()

    hidden_pre = torch.nan_to_num(F.linear(x, global_in_w, global_in_b), nan=0.0, posinf=0.0, neginf=0.0)
    hidden = torch.nan_to_num(act_fn(hidden_pre), nan=0.0, posinf=0.0, neginf=0.0)
    global_mlp_output = torch.nan_to_num(F.linear(hidden, global_out_w, global_out_b), nan=0.0, posinf=0.0, neginf=0.0)

    all_coords = torch.nan_to_num(torch.matmul(x, input_basis), nan=0.0, posinf=0.0, neginf=0.0)
    all_coords = all_coords.reshape(batch_tokens, num_subspaces, subspace_rank)

    shifted = all_coords - centers
    dist_sq = shifted.square().sum(dim=2)
    routing_scores = -dist_sq
    topk_scores, topk_indices = torch.topk(routing_scores, k=topk_neighbors, dim=1, sorted=False)
    temp = torch.as_tensor(routing_temperature * ROUTING_TEMP_SCALE, device=device, dtype=topk_scores.dtype)
    routing_weights = torch.softmax(topk_scores / temp, dim=1).to(dtype=compute_dtype)

    expanded = topk_indices.unsqueeze(-1).expand(-1, -1, subspace_rank)
    coords_topk = all_coords.gather(1, expanded)
    flat_indices = topk_indices.reshape(-1)

    def _gather_subspace(tensor: Tensor) -> Tensor:
        gathered = tensor.index_select(0, flat_indices)
        return gathered.reshape(topk_indices.shape[0], topk_indices.shape[1], *tensor.shape[1:])

    anchor_start_topk = _gather_subspace(anchor_start)
    anchor_end_topk = _gather_subspace(anchor_end)
    basis_start_topk = _gather_subspace(basis_start)
    basis_end_topk = _gather_subspace(basis_end)
    pos_tangent_start_topk = _gather_subspace(pos_tangent_start)
    pos_tangent_end_topk = _gather_subspace(pos_tangent_end)
    basis_tangent_start_topk = _gather_subspace(basis_tangent_start)
    basis_tangent_end_topk = _gather_subspace(basis_tangent_end)
    output_basis_topk = _gather_subspace(output_basis)
    sigma_topk = _gather_subspace(sigma)
    extrapolation_topk = _gather_subspace(extrapolation)

    local_w_in_topk = _gather_subspace(local_w_in)
    local_b_in_topk = _gather_subspace(local_b_in)
    local_w_out_topk = _gather_subspace(local_w_out)
    local_b_out_topk = _gather_subspace(local_b_out)
    local_w_gate_topk = _gather_subspace(local_w_gate)
    local_b_gate_topk = _gather_subspace(local_b_gate)

    coords_rep = coords_topk.unsqueeze(2).expand(-1, -1, num_splanifolds, -1)
    flat_coords = coords_rep.reshape(-1, subspace_rank)
    y_splanifold = k_splanifold_eval(
        flat_coords,
        anchor_start_topk.reshape(-1, subspace_rank),
        anchor_end_topk.reshape(-1, subspace_rank),
        basis_start_topk.reshape(-1, subspace_rank, subspace_rank),
        basis_end_topk.reshape(-1, subspace_rank, subspace_rank),
        pos_tangent_start_topk.reshape(-1, subspace_rank, subspace_rank),
        pos_tangent_end_topk.reshape(-1, subspace_rank, subspace_rank),
        basis_tangent_start_topk.reshape(-1, subspace_rank, subspace_rank),
        basis_tangent_end_topk.reshape(-1, subspace_rank, subspace_rank),
        sigma_topk.reshape(-1),
        extrapolation_topk.reshape(-1),
    ).reshape(batch_tokens, topk_neighbors, num_splanifolds, subspace_rank)

    flat_topk = batch_tokens * topk_neighbors
    coords_flat = coords_topk.reshape(flat_topk, subspace_rank, 1)
    local_w_in_flat = local_w_in_topk.reshape(flat_topk, local_hidden, subspace_rank)
    local_b_in_flat = local_b_in_topk.reshape(flat_topk, local_hidden)
    local_w_out_flat = local_w_out_topk.reshape(flat_topk, subspace_rank, local_hidden)
    local_b_out_flat = local_b_out_topk.reshape(flat_topk, subspace_rank)
    local_w_gate_flat = local_w_gate_topk.reshape(flat_topk, num_splanifolds + 1, local_hidden)
    local_b_gate_flat = local_b_gate_topk.reshape(flat_topk, num_splanifolds + 1)

    hidden_local = torch.bmm(local_w_in_flat, coords_flat).squeeze(-1) + local_b_in_flat
    hidden_local = torch.nan_to_num(act_fn(hidden_local), nan=0.0, posinf=0.0, neginf=0.0)
    local_out = torch.bmm(local_w_out_flat, hidden_local.unsqueeze(-1)).squeeze(-1) + local_b_out_flat
    gate_logits = torch.bmm(local_w_gate_flat, hidden_local.unsqueeze(-1)).squeeze(-1) + local_b_gate_flat
    gate_weights = torch.softmax(gate_logits, dim=1)

    local_out = local_out.reshape(batch_tokens, topk_neighbors, subspace_rank)
    gate_weights = gate_weights.reshape(batch_tokens, topk_neighbors, num_splanifolds + 1)
    bundle_outputs = torch.cat([y_splanifold, local_out.unsqueeze(2)], dim=2)
    y_sub = (bundle_outputs * gate_weights.unsqueeze(-1)).sum(dim=2)

    y_sub_flat = y_sub.reshape(flat_topk, subspace_rank)
    output_basis_flat = output_basis_topk.reshape(flat_topk, subspace_rank, output_dim)
    projected = torch.bmm(y_sub_flat.unsqueeze(1), output_basis_flat.to(dtype=y_sub_flat.dtype)).squeeze(1)
    projected = projected.reshape(batch_tokens, topk_neighbors, output_dim)
    subspace_output = (projected * routing_weights.unsqueeze(-1)).sum(dim=1)

    if subspace_output.dtype != compute_dtype:
        subspace_output = subspace_output.to(dtype=compute_dtype)

    final_output = global_mlp_output + subspace_output
    final_output = torch.nan_to_num(final_output, nan=0.0, posinf=0.0, neginf=0.0)
    final_output = torch.nan_to_num(final_output.to(dtype=input_batch.dtype), nan=0.0, posinf=0.0, neginf=0.0)
    return final_output


@triton.autotune(configs=_SSINF_SSM_FUSED_CONFIGS, key=["output_dim"])
@triton.jit
def _ssinf3_ssm_fused_kernel(
    token_idx_ptr,
    subspace_idx_ptr,
    routing_ptr,
    coords_ptr,
    anchor_start_ptr,
    anchor_end_ptr,
    basis_start_ptr,
    basis_end_ptr,
    pos_tangent_start_ptr,
    pos_tangent_end_ptr,
    basis_tangent_start_ptr,
    basis_tangent_end_ptr,
    sigma_ptr,
    extrap_ptr,
    local_w_in_ptr,
    local_b_in_ptr,
    local_w_out_ptr,
    local_b_out_ptr,
    local_w_gate_ptr,
    local_b_gate_ptr,
    output_basis_ptr,
    out_ptr,
    stride_token,
    stride_subspace,
    stride_routing,
    num_tokens,
    num_subspaces,
    stride_coords0,
    stride_coords1,
    stride_coords2,
    stride_anchor0,
    stride_anchor1,
    stride_anchor2,
    stride_basis0,
    stride_basis1,
    stride_basis2,
    stride_basis3,
    stride_pos0,
    stride_pos1,
    stride_pos2,
    stride_pos3,
    stride_bt0,
    stride_bt1,
    stride_bt2,
    stride_bt3,
    stride_sigma0,
    stride_sigma1,
    stride_extrap0,
    stride_extrap1,
    stride_lwi0,
    stride_lwi1,
    stride_lwi2,
    stride_lbi0,
    stride_lbi1,
    stride_lwo0,
    stride_lwo1,
    stride_lwo2,
    stride_lbo0,
    stride_lbo1,
    stride_lwg0,
    stride_lwg1,
    stride_lwg2,
    stride_lbg0,
    stride_lbg1,
    stride_out_basis0,
    stride_out_basis1,
    stride_out_basis2,
    stride_out0,
    stride_out1,
    output_dim: tl.constexpr,
    rank: tl.constexpr,
    num_splanifolds: tl.constexpr,
    num_gates: tl.constexpr,
    local_hidden: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    token = tl.load(token_idx_ptr + pid * stride_token).to(tl.int32)
    s = tl.load(subspace_idx_ptr + pid * stride_subspace).to(tl.int32)
    weight = tl.load(routing_ptr + pid * stride_routing).to(tl.float32)
    valid = (token >= 0) & (token < num_tokens) & (s >= 0) & (s < num_subspaces)
    token = tl.where(valid, token, 0)
    s = tl.where(valid, s, 0)
    weight = tl.where(valid, weight, 0.0)

    r_offsets = tl.arange(0, rank)
    r_mask = r_offsets < rank

    coords_ptrs = coords_ptr + token * stride_coords0 + s * stride_coords1 + r_offsets * stride_coords2
    coords = tl.load(coords_ptrs, mask=r_mask, other=0.0).to(tl.float32)

    GATE_BLOCK: tl.constexpr = 8
    g_offsets = tl.arange(0, GATE_BLOCK)
    g_mask = g_offsets < num_gates

    local_out = tl.load(
        local_b_out_ptr + s * stride_lbo0 + r_offsets * stride_lbo1, mask=r_mask, other=0.0
    ).to(tl.float32)
    gate_acc = tl.load(local_b_gate_ptr + s * stride_lbg0 + g_offsets * stride_lbg1, mask=g_mask, other=0.0).to(
        tl.float32
    )

    for h in tl.static_range(0, local_hidden, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < local_hidden

        w_in_ptrs = (
            local_w_in_ptr
            + s * stride_lwi0
            + h_offsets[:, None] * stride_lwi1
            + r_offsets[None, :] * stride_lwi2
        )
        w_in = tl.load(w_in_ptrs, mask=h_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
        pre = tl.sum(w_in * coords[None, :], axis=1)
        b_in = tl.load(local_b_in_ptr + s * stride_lbi0 + h_offsets * stride_lbi1, mask=h_mask, other=0.0).to(
            tl.float32
        )
        pre = pre + b_in
        act = 0.5 * pre * (1.0 + tl.erf(pre * 0.7071067811865476))

        w_out_ptrs = (
            local_w_out_ptr
            + s * stride_lwo0
            + r_offsets[:, None] * stride_lwo1
            + h_offsets[None, :] * stride_lwo2
        )
        w_out = tl.load(w_out_ptrs, mask=r_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        local_out += tl.sum(w_out * act[None, :], axis=1)

        w_gate_ptrs = (
            local_w_gate_ptr
            + s * stride_lwg0
            + g_offsets[:, None] * stride_lwg1
            + h_offsets[None, :] * stride_lwg2
        )
        w_gate = tl.load(w_gate_ptrs, mask=g_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        gate_acc += tl.sum(w_gate * act[None, :], axis=1)

    gate_acc = tl.where(g_mask, gate_acc, -float("inf"))
    g_max = tl.max(gate_acc, axis=0)
    g_exp = tl.exp(gate_acc - g_max)
    g_sum = tl.sum(g_exp, axis=0)
    gate_weights = g_exp / g_sum

    inv_rank = 1.0 / rank
    acc = tl.zeros([rank], dtype=tl.float32)
    r_row = r_offsets[:, None]
    r_col = r_offsets[None, :]
    r_mask_2d = (r_row < rank) & (r_col < rank)

    for m in tl.static_range(0, num_splanifolds):
        gate = tl.sum(gate_weights * (g_offsets == m), axis=0)
        sigma = tl.load(sigma_ptr + s * stride_sigma0 + m * stride_sigma1).to(tl.float32)
        extrap = tl.load(extrap_ptr + s * stride_extrap0 + m * stride_extrap1).to(tl.float32)

        u = coords * (1.0 + 2.0 * extrap) - extrap
        sum_u = tl.sum(u, axis=0)
        sum_u_abs = tl.abs(sum_u)
        is_small = sum_u_abs < 1e-4
        sum_u_safe = tl.where(is_small, 1.0, sum_u)
        w = u / sum_u_safe
        w = tl.where(is_small, inv_rank, w)

        t = sum_u * inv_rank
        delta = u - t

        anchor_start_ptrs = anchor_start_ptr + s * stride_anchor0 + m * stride_anchor1 + r_offsets * stride_anchor2
        anchor_end_ptrs = anchor_end_ptr + s * stride_anchor0 + m * stride_anchor1 + r_offsets * stride_anchor2
        anchor_start = tl.load(anchor_start_ptrs, mask=r_mask, other=0.0).to(tl.float32)
        anchor_end = tl.load(anchor_end_ptrs, mask=r_mask, other=0.0).to(tl.float32)

        basis_start_ptrs = (
            basis_start_ptr
            + s * stride_basis0
            + m * stride_basis1
            + r_row * stride_basis2
            + r_col * stride_basis3
        )
        basis_end_ptrs = (
            basis_end_ptr
            + s * stride_basis0
            + m * stride_basis1
            + r_row * stride_basis2
            + r_col * stride_basis3
        )
        basis_start = tl.load(basis_start_ptrs, mask=r_mask_2d, other=0.0).to(tl.float32)
        basis_end = tl.load(basis_end_ptrs, mask=r_mask_2d, other=0.0).to(tl.float32)

        pos_start_ptrs = (
            pos_tangent_start_ptr
            + s * stride_pos0
            + m * stride_pos1
            + r_row * stride_pos2
            + r_col * stride_pos3
        )
        pos_end_ptrs = (
            pos_tangent_end_ptr
            + s * stride_pos0
            + m * stride_pos1
            + r_row * stride_pos2
            + r_col * stride_pos3
        )
        bt_start_ptrs = (
            basis_tangent_start_ptr
            + s * stride_bt0
            + m * stride_bt1
            + r_row * stride_bt2
            + r_col * stride_bt3
        )
        bt_end_ptrs = (
            basis_tangent_end_ptr
            + s * stride_bt0
            + m * stride_bt1
            + r_row * stride_bt2
            + r_col * stride_bt3
        )

        pos_start = tl.load(pos_start_ptrs, mask=r_mask_2d, other=0.0).to(tl.float32)
        pos_end = tl.load(pos_end_ptrs, mask=r_mask_2d, other=0.0).to(tl.float32)
        bt_start = tl.load(bt_start_ptrs, mask=r_mask_2d, other=0.0).to(tl.float32)
        bt_end = tl.load(bt_end_ptrs, mask=r_mask_2d, other=0.0).to(tl.float32)

        v_start = tl.sum(pos_start * w[:, None], axis=0)
        v_end = tl.sum(pos_end * w[:, None], axis=0)
        anchor_start_dot = tl.sum(anchor_start * w, axis=0)
        anchor_end_dot = tl.sum(anchor_end * w, axis=0)
        v_start = v_start - anchor_start_dot
        v_end = v_end - anchor_end_dot

        d_start = tl.sum(basis_start * delta[:, None], axis=0)
        d_end = tl.sum(basis_end * delta[:, None], axis=0)
        t_start_pre = tl.sum((bt_start - basis_start) * delta[:, None], axis=0)
        t_end_pre = tl.sum((bt_end - basis_end) * delta[:, None], axis=0)

        v_start = sigma * v_start
        v_end = sigma * v_end
        t_start = sigma * t_start_pre
        t_end = sigma * t_end_pre

        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        spine = h00 * anchor_start + h01 * anchor_end + h10 * v_start + h11 * v_end
        disp = h00 * d_start + h01 * d_end + h10 * t_start + h11 * t_end
        y_sub = spine + disp

        acc += gate * y_sub

    gate_local = tl.sum(gate_weights * (g_offsets == (num_gates - 1)), axis=0)
    acc += gate_local * local_out

    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < output_dim
    col_mask = col_mask & valid
    basis_ptrs = (
        output_basis_ptr
        + s * stride_out_basis0
        + r_offsets[:, None] * stride_out_basis1
        + col_offsets[None, :] * stride_out_basis2
    )
    basis_vals = tl.load(basis_ptrs, mask=r_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    proj = tl.dot(acc[None, :], basis_vals)
    proj = tl.reshape(proj, (BLOCK_N,)).to(tl.float32)
    proj *= weight

    out_ptrs = out_ptr + token * stride_out0 + col_offsets * stride_out1
    tl.atomic_add(out_ptrs, proj, mask=col_mask)


def ssinf3_triton_forward_ssm_fused(
    input_batch: Tensor,
    input_basis_matrix: Tensor,
    center_projection: Tensor,
    output_basis: Tensor,
    splanifold_anchor_start: Tensor,
    splanifold_anchor_end: Tensor,
    splanifold_basis_start: Tensor,
    splanifold_basis_end: Tensor,
    splanifold_pos_tangent_start: Tensor,
    splanifold_pos_tangent_end: Tensor,
    splanifold_basis_tangent_start: Tensor,
    splanifold_basis_tangent_end: Tensor,
    splanifold_sigma: Tensor,
    splanifold_extrapolation: Tensor,
    local_mlp_weight_in: Tensor,
    local_mlp_bias_in: Tensor,
    local_mlp_weight_out: Tensor,
    local_mlp_bias_out: Tensor,
    local_mlp_weight_gate: Tensor,
    local_mlp_bias_gate: Tensor,
    global_mlp_weight_in: Tensor,
    global_mlp_bias_in: Tensor,
    global_mlp_weight_out: Tensor,
    global_mlp_bias_out: Tensor,
    topk_neighbors: int,
    routing_temperature: float,
    activation: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    if input_batch.ndim != 2:
        raise ValueError(f"input_batch must be 2D, got {tuple(input_batch.shape)}")
    if not input_batch.is_cuda:
        raise ValueError("SSINF3 Triton path requires CUDA/HIP tensors (input_batch.is_cuda=False).")

    routing_temperature = float(routing_temperature)
    if routing_temperature <= 0:
        raise ValueError("routing_temperature must be > 0.")

    device = input_batch.device
    weight_dtype = input_basis_matrix.dtype
    compute_dtype = _compute_dtype(weight_dtype)

    num_subspaces = int(center_projection.shape[0])
    topk_neighbors = int(topk_neighbors)
    if topk_neighbors <= 0:
        raise ValueError("topk_neighbors must be positive.")
    if topk_neighbors > num_subspaces:
        raise ValueError(f"topk_neighbors ({topk_neighbors}) must be <= num_subspaces ({num_subspaces}).")

    act_fn = activation or (lambda x: F.gelu(x, approximate="tanh"))

    x = input_batch.to(dtype=compute_dtype)
    input_basis = input_basis_matrix.to(dtype=compute_dtype, device=device)
    centers = center_projection.to(dtype=compute_dtype, device=device)
    output_basis = output_basis.to(dtype=compute_dtype, device=device)

    anchor_start = splanifold_anchor_start.to(dtype=compute_dtype, device=device)
    anchor_end = splanifold_anchor_end.to(dtype=compute_dtype, device=device)
    basis_start = splanifold_basis_start.to(dtype=compute_dtype, device=device)
    basis_end = splanifold_basis_end.to(dtype=compute_dtype, device=device)
    pos_tangent_start = splanifold_pos_tangent_start.to(dtype=compute_dtype, device=device)
    pos_tangent_end = splanifold_pos_tangent_end.to(dtype=compute_dtype, device=device)
    basis_tangent_start = splanifold_basis_tangent_start.to(dtype=compute_dtype, device=device)
    basis_tangent_end = splanifold_basis_tangent_end.to(dtype=compute_dtype, device=device)
    sigma = splanifold_sigma.to(dtype=compute_dtype, device=device)
    extrapolation = splanifold_extrapolation.to(dtype=compute_dtype, device=device)

    local_w_in = local_mlp_weight_in.to(dtype=compute_dtype, device=device)
    local_b_in = local_mlp_bias_in.to(dtype=compute_dtype, device=device)
    local_w_out = local_mlp_weight_out.to(dtype=compute_dtype, device=device)
    local_b_out = local_mlp_bias_out.to(dtype=compute_dtype, device=device)
    local_w_gate = local_mlp_weight_gate.to(dtype=compute_dtype, device=device)
    local_b_gate = local_mlp_bias_gate.to(dtype=compute_dtype, device=device)

    global_in_w = global_mlp_weight_in.to(dtype=compute_dtype, device=device)
    global_in_b = global_mlp_bias_in.to(dtype=compute_dtype, device=device)
    global_out_w = global_mlp_weight_out.to(dtype=compute_dtype, device=device)
    global_out_b = global_mlp_bias_out.to(dtype=compute_dtype, device=device)

    hidden_pre = F.linear(x, global_in_w, global_in_b)
    hidden = act_fn(hidden_pre)
    global_mlp_output = F.linear(hidden, global_out_w, global_out_b)

    all_coords = torch.mm(x, input_basis)
    batch_tokens = all_coords.shape[0]
    rank = centers.shape[1]
    all_coords = all_coords.reshape(batch_tokens, num_subspaces, rank)

    shifted = all_coords - centers
    dist_sq = shifted.square().sum(dim=2)
    temp = routing_temperature * ROUTING_TEMP_SCALE
    topk_scores, topk_idx = torch.topk(dist_sq, k=topk_neighbors, dim=1, largest=False, sorted=False)
    topk_scores = topk_scores.neg_()
    routing_weights = torch.softmax(topk_scores / temp, dim=1)

    flat_subspace = topk_idx.reshape(-1)
    flat_weight = routing_weights.reshape(-1).to(dtype=compute_dtype)
    flat_tokens = torch.arange(batch_tokens, device=device, dtype=flat_subspace.dtype).repeat_interleave(topk_neighbors)

    order = torch.argsort(flat_subspace)
    flat_subspace = flat_subspace[order]
    flat_tokens = flat_tokens[order]
    flat_weight = flat_weight[order]

    flat_subspace_i32 = flat_subspace.to(dtype=torch.int32)
    flat_tokens_i32 = flat_tokens.to(dtype=torch.int32)
    num_assignments = flat_subspace_i32.numel()

    out = torch.zeros((batch_tokens, output_basis.shape[2]), device=device, dtype=compute_dtype)

    grid = lambda meta: (num_assignments, triton.cdiv(out.shape[1], meta["BLOCK_N"]))
    _ssinf3_ssm_fused_kernel[grid](
        flat_tokens_i32,
        flat_subspace_i32,
        flat_weight,
        all_coords,
        anchor_start,
        anchor_end,
        basis_start,
        basis_end,
        pos_tangent_start,
        pos_tangent_end,
        basis_tangent_start,
        basis_tangent_end,
        sigma,
        extrapolation,
        local_w_in,
        local_b_in,
        local_w_out,
        local_b_out,
        local_w_gate,
        local_b_gate,
        output_basis,
        out,
        flat_tokens_i32.stride(0),
        flat_subspace_i32.stride(0),
        flat_weight.stride(0),
        batch_tokens,
        num_subspaces,
        all_coords.stride(0),
        all_coords.stride(1),
        all_coords.stride(2),
        anchor_start.stride(0),
        anchor_start.stride(1),
        anchor_start.stride(2),
        basis_start.stride(0),
        basis_start.stride(1),
        basis_start.stride(2),
        basis_start.stride(3),
        pos_tangent_start.stride(0),
        pos_tangent_start.stride(1),
        pos_tangent_start.stride(2),
        pos_tangent_start.stride(3),
        basis_tangent_start.stride(0),
        basis_tangent_start.stride(1),
        basis_tangent_start.stride(2),
        basis_tangent_start.stride(3),
        sigma.stride(0),
        sigma.stride(1),
        extrapolation.stride(0),
        extrapolation.stride(1),
        local_w_in.stride(0),
        local_w_in.stride(1),
        local_w_in.stride(2),
        local_b_in.stride(0),
        local_b_in.stride(1),
        local_w_out.stride(0),
        local_w_out.stride(1),
        local_w_out.stride(2),
        local_b_out.stride(0),
        local_b_out.stride(1),
        local_w_gate.stride(0),
        local_w_gate.stride(1),
        local_w_gate.stride(2),
        local_b_gate.stride(0),
        local_b_gate.stride(1),
        output_basis.stride(0),
        output_basis.stride(1),
        output_basis.stride(2),
        out.stride(0),
        out.stride(1),
        output_dim=out.shape[1],
        rank=rank,
        num_splanifolds=int(anchor_start.shape[1]),
        num_gates=int(anchor_start.shape[1]) + 1,
        local_hidden=local_w_in.shape[1],
        BLOCK_H=128,
    )

    out = out + global_mlp_output.to(dtype=compute_dtype)
    final_output = out
    if final_output.dtype != input_batch.dtype:
        final_output = final_output.to(dtype=input_batch.dtype)
    return final_output


@dataclass(eq=False)
class SSINF3Layer(nn.Module):
    in_features: int
    out_features: int
    num_subspaces: int = NUM_SUBSPACES
    subspace_rank: int = SUBSPACE_RANK
    num_splanifolds: int = NUM_SPLANIFOLDS
    local_hidden: int = LOCAL_MLP_HIDDEN_DIM
    global_hidden: int = GLOBAL_MLP_HIDDEN_DIM
    topk_neighbors: int = TOPK_NEIGHBORS
    routing_temperature: float = 1.0
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None
    splanifold_sigma_init: float = SPLANIFOLD_SIGMA
    splanifold_extrapolation_init: float = SPLANIFOLD_EXTRAPOLATION
    dtype: torch.dtype = WEIGHT_DTYPE
    device: torch.device | None = None
    use_fused_eval_only: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        weight_dtype = self.dtype
        all_rank = self.num_subspaces * self.subspace_rank
        self.input_basis_matrix = nn.Parameter(
            torch.randn(self.in_features, all_rank, device=self.device, dtype=weight_dtype) * 0.005
        )
        self.center_projection = nn.Parameter(
            torch.zeros(self.num_subspaces, self.subspace_rank, device=self.device, dtype=weight_dtype)
        )
        self.output_basis = nn.Parameter(
            torch.randn(self.num_subspaces, self.subspace_rank, self.out_features, device=self.device, dtype=weight_dtype)
            * 0.005
        )

        eye = torch.eye(self.subspace_rank, device=self.device, dtype=weight_dtype)
        basis_seed = eye.unsqueeze(0).unsqueeze(0).repeat(self.num_subspaces, self.num_splanifolds, 1, 1)
        basis_seed = basis_seed + torch.randn_like(basis_seed) * 0.0025

        self.splanifold_anchor_start = nn.Parameter(
            torch.zeros(self.num_subspaces, self.num_splanifolds, self.subspace_rank, device=self.device, dtype=weight_dtype)
        )
        self.splanifold_anchor_end = nn.Parameter(
            torch.zeros(self.num_subspaces, self.num_splanifolds, self.subspace_rank, device=self.device, dtype=weight_dtype)
        )
        self.splanifold_basis_start = nn.Parameter(basis_seed.clone())
        self.splanifold_basis_end = nn.Parameter(basis_seed.clone())
        self.splanifold_pos_tangent_start = nn.Parameter(
            torch.zeros(
                self.num_subspaces,
                self.num_splanifolds,
                self.subspace_rank,
                self.subspace_rank,
                device=self.device,
                dtype=weight_dtype,
            )
        )
        self.splanifold_pos_tangent_end = nn.Parameter(
            torch.zeros(
                self.num_subspaces,
                self.num_splanifolds,
                self.subspace_rank,
                self.subspace_rank,
                device=self.device,
                dtype=weight_dtype,
            )
        )
        self.splanifold_basis_tangent_start = nn.Parameter(
            torch.zeros(
                self.num_subspaces,
                self.num_splanifolds,
                self.subspace_rank,
                self.subspace_rank,
                device=self.device,
                dtype=weight_dtype,
            )
        )
        self.splanifold_basis_tangent_end = nn.Parameter(
            torch.zeros(
                self.num_subspaces,
                self.num_splanifolds,
                self.subspace_rank,
                self.subspace_rank,
                device=self.device,
                dtype=weight_dtype,
            )
        )

        self._sigma_min = 1e-4
        self._sigma_max = 10.0
        self._extrap_max = 0.5

        sigma_init = float(self.splanifold_sigma_init)
        sigma_init = min(max(sigma_init, self._sigma_min), self._sigma_max)
        sigma_init_t = torch.full(
            (self.num_subspaces, self.num_splanifolds),
            sigma_init - self._sigma_min,
            device=self.device,
            dtype=torch.float32,
        )
        sigma_raw = _inv_softplus(sigma_init_t)
        self.splanifold_sigma_raw = nn.Parameter(sigma_raw.to(dtype=weight_dtype))

        extrap_init = float(self.splanifold_extrapolation_init)
        extrap_limit = self._extrap_max - 1e-4
        extrap_init = min(max(extrap_init, -extrap_limit), extrap_limit)
        extrap_norm = extrap_init / self._extrap_max
        extrap_raw = _inv_tanh(
            torch.full(
                (self.num_subspaces, self.num_splanifolds),
                extrap_norm,
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.splanifold_extrapolation_raw = nn.Parameter(extrap_raw.to(dtype=weight_dtype))

        self.local_mlp_weight_in = nn.Parameter(
            torch.randn(self.num_subspaces, self.local_hidden, self.subspace_rank, device=self.device, dtype=weight_dtype)
            * 0.01
        )
        self.local_mlp_bias_in = nn.Parameter(
            torch.zeros(self.num_subspaces, self.local_hidden, device=self.device, dtype=weight_dtype)
        )
        self.local_mlp_weight_out = nn.Parameter(
            torch.randn(self.num_subspaces, self.subspace_rank, self.local_hidden, device=self.device, dtype=weight_dtype)
            * 0.01
        )
        self.local_mlp_bias_out = nn.Parameter(
            torch.zeros(self.num_subspaces, self.subspace_rank, device=self.device, dtype=weight_dtype)
        )
        self.local_mlp_weight_gate = nn.Parameter(
            torch.randn(
                self.num_subspaces,
                self.num_splanifolds + 1,
                self.local_hidden,
                device=self.device,
                dtype=weight_dtype,
            )
            * 0.01
        )
        self.local_mlp_bias_gate = nn.Parameter(
            torch.zeros(self.num_subspaces, self.num_splanifolds + 1, device=self.device, dtype=weight_dtype)
        )

        self.global_mlp_weight_in = nn.Parameter(
            torch.randn(self.global_hidden, self.in_features, device=self.device, dtype=weight_dtype) * 0.005
        )
        self.global_mlp_bias_in = nn.Parameter(torch.zeros(self.global_hidden, device=self.device, dtype=weight_dtype))
        self.global_mlp_weight_out = nn.Parameter(
            torch.randn(self.out_features, self.global_hidden, device=self.device, dtype=weight_dtype) * 0.005
        )
        self.global_mlp_bias_out = nn.Parameter(torch.zeros(self.out_features, device=self.device, dtype=weight_dtype))

        def _tag_params(params, group: str) -> None:
            for param in params:
                param.label = "ssinf"
                param.ssinf_group = group

        _tag_params((self.input_basis_matrix, self.output_basis, self.splanifold_basis_start, self.splanifold_basis_end), "basis")
        _tag_params((self.center_projection, self.splanifold_anchor_start, self.splanifold_anchor_end), "anchors")
        _tag_params(
            (
                self.splanifold_pos_tangent_start,
                self.splanifold_pos_tangent_end,
                self.splanifold_basis_tangent_start,
                self.splanifold_basis_tangent_end,
            ),
            "tangents",
        )
        _tag_params((self.splanifold_sigma_raw, self.splanifold_extrapolation_raw), "scalars")
        _tag_params(
            (
                self.local_mlp_weight_in,
                self.local_mlp_bias_in,
                self.local_mlp_weight_out,
                self.local_mlp_bias_out,
                self.local_mlp_weight_gate,
                self.local_mlp_bias_gate,
            ),
            "local_mlp",
        )
        _tag_params(
            (
                self.global_mlp_weight_in,
                self.global_mlp_bias_in,
                self.global_mlp_weight_out,
                self.global_mlp_bias_out,
            ),
            "global_mlp",
        )

    @property
    def splanifold_sigma(self) -> torch.Tensor:
        return _compute_sigma(self.splanifold_sigma_raw, self._sigma_min, self._sigma_max)

    @property
    def splanifold_extrapolation(self) -> torch.Tensor:
        return _compute_extrapolation(self.splanifold_extrapolation_raw, self._extrap_max)

    def _flatten(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if x.dim() == 2:
            return x, None
        if x.dim() == 3:
            bsz, seq, _ = x.shape
            return x.reshape(bsz * seq, x.shape[-1]), (bsz, seq)
        raise ValueError("SSINF3Layer expects a [N, C] or [B, T, C] input tensor.")

    def _unflatten(self, x: torch.Tensor, shape: tuple[int, int] | None) -> torch.Tensor:
        if shape is None:
            return x
        bsz, seq = shape
        return x.reshape(bsz, seq, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat, shape = self._flatten(x)
        if self.use_fused_eval_only and not self.training:
            y = ssinf3_triton_forward_ssm_fused(
                x_flat,
                input_basis_matrix=self.input_basis_matrix,
                center_projection=self.center_projection,
                output_basis=self.output_basis,
                splanifold_anchor_start=self.splanifold_anchor_start,
                splanifold_anchor_end=self.splanifold_anchor_end,
                splanifold_basis_start=self.splanifold_basis_start,
                splanifold_basis_end=self.splanifold_basis_end,
                splanifold_pos_tangent_start=self.splanifold_pos_tangent_start,
                splanifold_pos_tangent_end=self.splanifold_pos_tangent_end,
                splanifold_basis_tangent_start=self.splanifold_basis_tangent_start,
                splanifold_basis_tangent_end=self.splanifold_basis_tangent_end,
                splanifold_sigma=self.splanifold_sigma,
                splanifold_extrapolation=self.splanifold_extrapolation,
                local_mlp_weight_in=self.local_mlp_weight_in,
                local_mlp_bias_in=self.local_mlp_bias_in,
                local_mlp_weight_out=self.local_mlp_weight_out,
                local_mlp_bias_out=self.local_mlp_bias_out,
                local_mlp_weight_gate=self.local_mlp_weight_gate,
                local_mlp_bias_gate=self.local_mlp_bias_gate,
                global_mlp_weight_in=self.global_mlp_weight_in,
                global_mlp_bias_in=self.global_mlp_bias_in,
                global_mlp_weight_out=self.global_mlp_weight_out,
                global_mlp_bias_out=self.global_mlp_bias_out,
                topk_neighbors=self.topk_neighbors,
                routing_temperature=self.routing_temperature,
                activation=self.activation,
            )
        else:
            y = ssinf3_torch_reference_forward(
                x_flat,
                input_basis_matrix=self.input_basis_matrix,
                center_projection=self.center_projection,
                output_basis=self.output_basis,
                splanifold_anchor_start=self.splanifold_anchor_start,
                splanifold_anchor_end=self.splanifold_anchor_end,
                splanifold_basis_start=self.splanifold_basis_start,
                splanifold_basis_end=self.splanifold_basis_end,
                splanifold_pos_tangent_start=self.splanifold_pos_tangent_start,
                splanifold_pos_tangent_end=self.splanifold_pos_tangent_end,
                splanifold_basis_tangent_start=self.splanifold_basis_tangent_start,
                splanifold_basis_tangent_end=self.splanifold_basis_tangent_end,
                splanifold_sigma=self.splanifold_sigma,
                splanifold_extrapolation=self.splanifold_extrapolation,
                local_mlp_weight_in=self.local_mlp_weight_in,
                local_mlp_bias_in=self.local_mlp_bias_in,
                local_mlp_weight_out=self.local_mlp_weight_out,
                local_mlp_bias_out=self.local_mlp_bias_out,
                local_mlp_weight_gate=self.local_mlp_weight_gate,
                local_mlp_bias_gate=self.local_mlp_bias_gate,
                global_mlp_weight_in=self.global_mlp_weight_in,
                global_mlp_bias_in=self.global_mlp_bias_in,
                global_mlp_weight_out=self.global_mlp_weight_out,
                global_mlp_bias_out=self.global_mlp_bias_out,
                topk_neighbors=self.topk_neighbors,
                routing_temperature=self.routing_temperature,
                activation=self.activation,
            )
        return self._unflatten(y, shape)


class SSINF3(nn.Module):
    # Triton SSM path only.
    def __init__(self, model_dim: int, output_dim: int | None = None, **kwargs) -> None:
        super().__init__()
        forbidden_keys = {
            "num_subspaces",
            "subspace_rank",
            "num_splanifolds",
            "local_hidden",
            "global_hidden",
            "topk_neighbors",
        }
        invalid_keys = forbidden_keys.intersection(kwargs.keys())
        if invalid_keys:
            invalid_keys_str = ", ".join(sorted(invalid_keys))
            raise TypeError(f"SSINF3 architecture is fixed by module constants; do not pass: {invalid_keys_str}")
        out_dim = output_dim if output_dim is not None else model_dim
        self.layer = SSINF3Layer(
            in_features=model_dim,
            out_features=out_dim,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, *_):
        return self.layer(x)
