import torch
import triton
import triton.language as tl

TOKEN_TILE_SIZE = 64
D_TILE_SIZE    = 64
V_TILE_SIZE    = 256

# ------------------------- 前向内核 -------------------------
@triton.jit
def fused_ce_fwd_kernel(
    hidden_ptr, weight_ptr, labels_ptr,
    loss_ptr, max_ptr, sum_ptr,
    stride_hb, stride_hs,
    stride_wv, stride_wd,
    stride_lblb, stride_lbls,
    B, S, D, V,
    TOKEN_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
    V_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_start = pid * TOKEN_TILE_SIZE
    token_offsets = token_start + tl.arange(0, TOKEN_TILE_SIZE)
    token_mask = token_offsets < (B * S)

    # 即使该 tile 无有效 token，也写回安全值，避免反向读取未初始化内存
    if tl.sum(token_mask) == 0:
        tl.store(loss_ptr + token_offsets, 0.0, mask=token_mask)
        tl.store(max_ptr + token_offsets, 0.0, mask=token_mask)
        tl.store(sum_ptr + token_offsets, 1.0, mask=token_mask)
        return

    # 不能用 block_size_ptr 吗？
    batch_ids = token_offsets // S
    seq_ids = token_offsets % S
    hid_offsets = batch_ids * stride_hb + seq_ids * stride_hs
    lbl_offsets = batch_ids * stride_lblb + seq_ids * stride_lbls
    # 越界 label 置为 0
    labels = tl.load(labels_ptr + lbl_offsets, mask=token_mask, other=0)

    # 在线 softmax 状态
    global_max = tl.full([TOKEN_TILE_SIZE], -float('inf'), dtype=tl.float32)
    global_sum = tl.zeros([TOKEN_TILE_SIZE], dtype=tl.float32)
    label_logit = tl.zeros([TOKEN_TILE_SIZE], dtype=tl.float32)

    for v_start in range(0, V, V_TILE_SIZE):
        v_offsets = v_start + tl.arange(0, V_TILE_SIZE)
        v_mask = v_offsets < V

        w_block_ptr = tl.make_block_ptr(
            base=weight_ptr,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(v_start, 0),
            block_shape=(V_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )

        acc = tl.zeros([TOKEN_TILE_SIZE, V_TILE_SIZE], dtype=tl.float32)

        # 内层循环：遍历 D 分块，累积完整 logits
        for d_start in range(0, D, D_TILE_SIZE):
            d_offsets = d_start + tl.arange(0, D_TILE_SIZE)
            d_mask = d_offsets < D
            x = tl.load(
                hidden_ptr + hid_offsets[:, None] + d_offsets[None, :],
                mask=(token_mask[:, None] & d_mask[None, :]),
                other=0.0
            )  # [TOKEN_TILE_SIZE, D_TILE_SIZE]
            w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")  # [V_TILE_SIZE, D_TILE_SIZE]
            acc += tl.dot(x.to(tl.float32), tl.trans(w).to(tl.float32))
            w_block_ptr.advance((0, D_TILE_SIZE))

        # 对于无效的 v 位置，acc 为 0，它们不影响 local_max，但为了安全强制屏蔽
        acc = tl.where(v_mask[None, :], acc, -float('inf'))  # 求 max 时忽略无效位
        local_max = tl.max(acc, axis=1)
        local_sum = tl.sum(tl.exp(acc - local_max[:, None]), axis=1)

        old_max = global_max
        global_max = tl.maximum(global_max, local_max)
        global_sum = (global_sum * tl.exp(old_max - global_max) +
                      local_sum * tl.exp(local_max - global_max))

        label_mask = (labels >= v_start) & (labels < v_start + V_TILE_SIZE)
        label_offset = labels - v_start
        oh_mask = (tl.arange(0, V_TILE_SIZE)[None, :] == label_offset[:, None])
        gathered = tl.sum(acc * oh_mask, axis=1)
        label_logit = tl.where(label_mask, gathered, label_logit)

    # 分块 logsumexp
    per_token_loss = -label_logit + global_max + tl.log(global_sum)
    per_token_loss = tl.where(token_mask, per_token_loss, 0.0)

    tl.store(loss_ptr + token_offsets, per_token_loss, mask=token_mask)
    tl.store(max_ptr + token_offsets, global_max, mask=token_mask)
    tl.store(sum_ptr + token_offsets, global_sum, mask=token_mask)

# ------------------------- dx 反向内核 -------------------------
@triton.jit
def fused_ce_bwd_dx_kernel(
    grad_scale,       
    hidden_ptr, weight_ptr, labels_ptr,
    max_ptr, sum_ptr,
    dhidden_ptr,
    stride_hb, stride_hs,
    stride_wv, stride_wd,
    stride_lblb, stride_lbls,
    B, S, D, V,
    TOKEN_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
    V_TILE_SIZE: tl.constexpr,
):
    """
    1D grid: 每个 block 负责一个 token tile，输出完整的 dx [TILE, D]。
    避免每个 d_tile 重复计算完整 logits。
    """
    pid = tl.program_id(0)
    token_start = pid * TOKEN_TILE_SIZE
    token_offsets = token_start + tl.arange(0, TOKEN_TILE_SIZE)
    token_mask = token_offsets < B * S

    if tl.sum(token_mask) == 0:
        return

    batch_ids = token_offsets // S
    seq_ids = token_offsets % S
    hid_offsets = batch_ids * stride_hb + seq_ids * stride_hs
    lbl_offsets = batch_ids * stride_lblb + seq_ids * stride_lbls

    labels = tl.load(labels_ptr + lbl_offsets, mask=token_mask, other=0)
    max_val = tl.load(max_ptr + token_offsets, mask=token_mask, other=0.0)
    sum_val = tl.load(sum_ptr + token_offsets, mask=token_mask, other=1.0)

    # 遍历 V 分块
    for v_start in range(0, V, V_TILE_SIZE):
        v_offsets = v_start + tl.arange(0, V_TILE_SIZE)
        v_mask = v_offsets < V

        w_block_ptr = tl.make_block_ptr(
            base=weight_ptr,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(v_start, 0),
            block_shape=(V_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )
        # 利用 max_val 和 sum_val 重计算 logits [D, V]
        logits = tl.zeros([TOKEN_TILE_SIZE, V_TILE_SIZE], dtype=tl.float32)

        for d_start in range(0, D, D_TILE_SIZE):
            d_offsets = d_start + tl.arange(0, D_TILE_SIZE)
            d_mask = d_offsets < D
            x = tl.load(
                hidden_ptr + hid_offsets[:, None] + d_offsets[None, :],
                mask=(token_mask[:, None] & d_mask[None, :]),
                other=0.0
            )
            w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")
            logits += tl.dot(x.to(tl.float32), tl.trans(w).to(tl.float32))
            w_block_ptr.advance((0, D_TILE_SIZE))

        # ---- 计算 softmax 概率 ----
        p = tl.exp(logits - max_val[:, None]) / sum_val[:, None]
        p = tl.where(v_mask[None, :], p, 0.0)
       
        label_mask = (labels >= v_start) & (labels < v_start + V_TILE_SIZE) & token_mask
        label_offset = labels - v_start
        oh_mask = (tl.arange(0, V_TILE_SIZE)[None, :] == label_offset[:, None])
        p = tl.where(label_mask[:, None] & oh_mask, p - 1.0, p)

        w_dx_ptr = tl.make_block_ptr(
            base=weight_ptr,
            shape=(V, D),
            strides=(stride_wv, stride_wd),
            offsets=(v_start, 0),
            block_shape=(V_TILE_SIZE, D_TILE_SIZE),
            order=(1, 0),
        )
        
        for d_start in range(0, D, D_TILE_SIZE):
            d_offsets = d_start + tl.arange(0, D_TILE_SIZE)
            d_mask = d_offsets < D
            w = tl.load(w_dx_ptr, boundary_check=(0, 1), padding_option="zero") # [V_TILE, D_TILE]
            dx_part = tl.dot(p.to(tl.float32), w.to(tl.float32))

            old = tl.load(
                dhidden_ptr + hid_offsets[:, None] + d_offsets[None, :],
                mask=(token_mask[:, None] & d_mask[None, :]),
                other=0.0
            )
            tl.store(
                dhidden_ptr + hid_offsets[:, None] + d_offsets[None, :],
                old + dx_part * grad_scale,
                mask=(token_mask[:, None] & d_mask[None, :])
            )
            w_dx_ptr.advance((0, D_TILE_SIZE))


# ------------------------- dw 反向内核 -------------------------
@triton.jit
def fused_ce_bwd_dw_kernel(
    grad_scale,    
    hidden_ptr, weight_ptr,
    labels_ptr, max_ptr, sum_ptr,
    dweight_ptr,
    stride_hb, stride_hs, 
    stride_wv, stride_wd,
    stride_lblb, stride_lbls,
    B, S, D, V,
    TOKEN_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
    V_TILE_SIZE: tl.constexpr,
):
    v_tile_idx = tl.program_id(0)
    d_tile_idx = tl.program_id(1)

    v_start = v_tile_idx * V_TILE_SIZE
    d_start = d_tile_idx * D_TILE_SIZE

    v_offsets = v_start + tl.arange(0, V_TILE_SIZE)
    v_mask = v_offsets < V
    d_offsets = d_start + tl.arange(0, D_TILE_SIZE)
    d_mask = d_offsets < D

    acc = tl.zeros([V_TILE_SIZE, D_TILE_SIZE], dtype=tl.float32)

    N = B * S
    for token_start in range(0, N, TOKEN_TILE_SIZE):
        token_offsets = token_start + tl.arange(0, TOKEN_TILE_SIZE)
        token_mask = token_offsets < N
        if tl.sum(token_mask) > 0:
            batch_ids = token_offsets // S
            seq_ids = token_offsets % S
            hid_offsets = batch_ids * stride_hb + seq_ids * stride_hs
            lbl_offsets = batch_ids * stride_lblb + seq_ids * stride_lbls

            labels = tl.load(labels_ptr + lbl_offsets, mask=token_mask, other=0)
            max_val = tl.load(max_ptr + token_offsets, mask=token_mask, other=0.0)
            sum_val = tl.load(sum_ptr + token_offsets, mask=token_mask, other=1.0)

            w_block_ptr = tl.make_block_ptr(
                base=weight_ptr,
                shape=(V, D),
                strides=(stride_wv, stride_wd),
                offsets=(v_start, 0),
                block_shape=(V_TILE_SIZE, D_TILE_SIZE),
                order=(1, 0),
            )
            # ---- 计算当前 V tile 在所有 token 上的完整 logits ----
            logits = tl.zeros([TOKEN_TILE_SIZE, V_TILE_SIZE], dtype=tl.float32)
            for d_inner in range(0, D, D_TILE_SIZE):
                d_inner_offsets = d_inner + tl.arange(0, D_TILE_SIZE)
                d_inner_mask = d_inner_offsets < D
                x = tl.load(
                    hidden_ptr + hid_offsets[:, None] + d_inner_offsets[None, :],
                    mask=(token_mask[:, None] & d_inner_mask[None, :]),
                    other=0.0
                )
                w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")
                logits += tl.dot(x.to(tl.float32), tl.trans(w).to(tl.float32))
                w_block_ptr.advance((0, D_TILE_SIZE))

            # ---- softmax 概率 ----
            p = tl.exp(logits - max_val[:, None]) / sum_val[:, None]
            p = tl.where(v_mask[None, :], p, 0.0)
            label_mask = (labels >= v_start) & (labels < v_start + V_TILE_SIZE) & token_mask
            label_offset = labels - v_start
            oh_mask = (tl.arange(0, V_TILE_SIZE)[None, :] == label_offset[:, None])
            p = tl.where(label_mask[:, None] & oh_mask, p - 1.0, p)

            # ---- 累加梯度到当前 d_tile ----
            x_d = tl.load(
                hidden_ptr + hid_offsets[:, None] + d_offsets[None, :],
                mask=(token_mask[:, None] & d_mask[None, :]),
                other=0.0
            )  # [TILE, D_TILE]
            acc += tl.dot(tl.trans(p.to(tl.float32)), x_d.to(tl.float32))

    # 写回
    dweight_block_ptr = tl.make_block_ptr(
        dweight_ptr,
        shape=(V, D),
        strides=(stride_wv, stride_wd),
        offsets=(v_start, d_start),
        block_shape=(V_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    tl.store(dweight_block_ptr, acc * grad_scale, boundary_check=(0, 1))

# ------------------------- PyTorch Autograd Function -------------------------
class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels):
        B, S, D = hidden.shape
        V, _ = weight.shape
        N = B * S

        loss_per_token = torch.empty(N, dtype=torch.float32, device=hidden.device)
        max_vals = torch.empty(N, dtype=torch.float32, device=hidden.device)
        sum_vals = torch.empty(N, dtype=torch.float32, device=hidden.device)

        grid = lambda meta: ((N + TOKEN_TILE_SIZE - 1) // TOKEN_TILE_SIZE,)
        fused_ce_fwd_kernel[grid](
            hidden, weight, labels,
            loss_per_token, max_vals, sum_vals,
            hidden.stride(0), hidden.stride(1), 
            weight.stride(0), weight.stride(1),
            labels.stride(0), labels.stride(1),
            B, S, D, V,
            TOKEN_TILE_SIZE, D_TILE_SIZE, V_TILE_SIZE
        )

        ctx.save_for_backward(hidden, weight, labels, max_vals, sum_vals)
        ctx.B, ctx.S, ctx.D, ctx.V, ctx.N = B, S, D, V, N
        return loss_per_token.sum() / N

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels, max_vals, sum_vals = ctx.saved_tensors
        B, S, D, V, N = ctx.B, ctx.S, ctx.D, ctx.V, ctx.N

        dhidden = torch.zeros_like(hidden)
        dweight = torch.zeros_like(weight)

        # 梯度缩放因子：因为 forward 返回 mean loss，需要除以 N
        grad_scale = grad_output.item() / N

        # dx 内核：1D grid，每个 block 输出一个 token tile 的完整 dx
        grid_dx = lambda meta: ((N + TOKEN_TILE_SIZE - 1) // TOKEN_TILE_SIZE,)
        fused_ce_bwd_dx_kernel[grid_dx](
            grad_scale,
            hidden, weight, labels, max_vals, sum_vals,
            dhidden,
            hidden.stride(0), hidden.stride(1),
            weight.stride(0), weight.stride(1),
            labels.stride(0), labels.stride(1),
            B, S, D, V,
            TOKEN_TILE_SIZE, D_TILE_SIZE, V_TILE_SIZE
        )

        # dw 内核：2D grid
        grid_dw = lambda meta: (
            (V + V_TILE_SIZE - 1) // V_TILE_SIZE,
            (D + D_TILE_SIZE - 1) // D_TILE_SIZE,
        )
        fused_ce_bwd_dw_kernel[grid_dw](
            grad_scale,
            hidden, weight, labels, max_vals, sum_vals,
            dweight,
            hidden.stride(0), hidden.stride(1),
            weight.stride(0), weight.stride(1),
            labels.stride(0), labels.stride(1),
            B, S, D, V,
            TOKEN_TILE_SIZE, D_TILE_SIZE, V_TILE_SIZE
        )

        return dhidden, dweight, None