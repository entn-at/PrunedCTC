# Copyright (c) 2026 Shanghai Jiao Tong University (author: Yifan Yang)
# SPDX-License-Identifier: MIT

"""Pruned CTC with full-vocabulary normalization and chunked recomputation."""

from __future__ import annotations

import importlib
import logging
import math
from numbers import Integral, Real
from types import ModuleType
from typing import TYPE_CHECKING, Any, overload

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

if TYPE_CHECKING:
    import k2

__version__ = "0.1.0"
__all__ = ["pruned_ctc_loss"]

# k2 stores dense FSA labels in uint16.
MAX_REDUCED_VOCAB = 65535
DEFAULT_MAX_STATES = 100_000_000

_FLOAT_DTYPES = (torch.float32, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)
_INT32_MAX = torch.iinfo(torch.int32).max
_LOGGER = logging.getLogger(__name__)
_warned_near_cap = False


def _load_k2() -> ModuleType:
    try:
        return importlib.import_module("k2")
    except (ImportError, OSError) as error:
        raise ImportError(
            "pruned_ctc_loss requires k2 built for your PyTorch, Python, and CUDA "
            "versions. Install a matching k2 build before using the loss: "
            "https://k2-fsa.github.io/k2/installation/index.html"
        ) from error


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return int(value)


def _validate_inputs(
    encoder_out: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    y: k2.RaggedTensor,
    encoder_out_lens: Tensor,
    blank_id: int,
    backend: ModuleType,
) -> Tensor:
    for name, value in (
        ("encoder_out", encoder_out),
        ("weight", weight),
        ("encoder_out_lens", encoder_out_lens),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
    if bias is not None and not isinstance(bias, Tensor):
        raise TypeError("bias must be a torch.Tensor or None.")
    if encoder_out.ndim != 3 or any(size == 0 for size in encoder_out.shape):
        raise ValueError("encoder_out must have nonempty shape (N, T, D).")
    if weight.ndim != 2 or weight.shape[0] == 0:
        raise ValueError("weight must have nonempty shape (V, D).")
    batch_size, max_frames, hidden_dim = encoder_out.shape
    vocab_size = weight.shape[0]
    if weight.shape[1] != hidden_dim:
        raise ValueError("weight and encoder_out must have the same hidden dimension.")
    if vocab_size > _INT32_MAX:
        raise ValueError("The vocabulary must fit in k2's signed int32 label indices.")
    if bias is not None and bias.shape != (vocab_size,):
        raise ValueError(f"bias must have shape ({vocab_size},).")
    device = encoder_out.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("pruned_ctc_loss supports CPU and CUDA tensors.")
    for name, value in (("encoder_out", encoder_out), ("weight", weight), ("bias", bias)):
        if value is None:
            continue
        if value.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"{name} must use float32 or bfloat16, got {value.dtype}.")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}.")
    if torch.is_autocast_enabled(device.type):
        raise RuntimeError(
            "pruned_ctc_loss controls its own precision. Call it inside "
            f"torch.autocast('{device.type}', enabled=False)."
        )
    if isinstance(blank_id, bool) or not isinstance(blank_id, Integral):
        raise TypeError("blank_id must be an integer.")
    if not 0 <= blank_id < vocab_size:
        raise ValueError(f"blank_id must be in [0, {vocab_size}), got {blank_id}.")
    if encoder_out_lens.ndim != 1 or encoder_out_lens.shape[0] != batch_size:
        raise ValueError(f"encoder_out_lens must have shape ({batch_size},).")
    if encoder_out_lens.dtype not in _INDEX_DTYPES:
        raise TypeError("encoder_out_lens must use int32 or int64.")
    if encoder_out_lens.device.type != "cpu" and encoder_out_lens.device != device:
        raise ValueError("encoder_out_lens must be on CPU or the encoder device.")
    lengths_cpu = encoder_out_lens.detach().to(device="cpu", dtype=torch.int64)
    lengths_cpu = lengths_cpu.clone(memory_format=torch.contiguous_format)
    if int(lengths_cpu.min()) < 1 or int(lengths_cpu.max()) > max_frames:
        raise ValueError(f"encoder_out_lens must contain values in [1, {max_frames}].")
    if int(lengths_cpu.sum()) > _INT32_MAX:
        raise ValueError("The number of valid frames exceeds k2's int32 index range.")
    if not isinstance(y, backend.RaggedTensor):
        raise TypeError("y must be a k2.RaggedTensor.")
    if y.num_axes != 2 or y.dim0 != batch_size:
        raise ValueError("y must have two axes and one target sequence per utterance.")
    if y.device != device:
        raise ValueError(f"y must be on {device}, got {y.device}.")
    targets = y.values
    if targets.dtype not in _INDEX_DTYPES:
        raise TypeError("Target IDs must use int32 or int64.")
    if targets.numel():
        invalid_range, contains_blank = torch.stack(
            (((targets < 0) | (targets >= vocab_size)).any(), (targets == blank_id).any())
        ).tolist()
        if invalid_range:
            raise ValueError(f"Target IDs must be in [0, {vocab_size}).")
        if contains_blank:
            raise ValueError("Target sequences must not contain blank_id.")
    return lengths_cpu


@overload
def _at_least_fp32(value: Tensor) -> Tensor: ...


@overload
def _at_least_fp32(value: None) -> None: ...


def _at_least_fp32(value: Tensor | None) -> Tensor | None:
    if value is None or value.dtype in (torch.float32, torch.float64):
        return value
    return value.float()


def _chunk_logits(hidden: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    if bias is None:
        return torch.mm(hidden, weight.t())
    return torch.addmm(bias, hidden, weight.t())


class _FusedReducedLogSoftmax(torch.autograd.Function):
    """Recompute dense softmax gradients in chunks; support first derivatives only."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        selected: Tensor,
        chunk_size: int = 4096,
    ) -> Tensor:
        hidden_dim = hidden.shape[-1]
        vocab_size = weight.shape[0]
        flat = _at_least_fp32(hidden.reshape(-1, hidden_dim))
        num_frames = flat.shape[0]
        dp_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32

        # Float64 normalization limits cancellation error in selected-class gradients.
        running_max = flat.new_full((num_frames,), -math.inf, dtype=torch.float64)
        running_sum = flat.new_zeros((num_frames,), dtype=torch.float64)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            logits = _chunk_logits(
                flat,
                _at_least_fp32(weight[start:end]),
                _at_least_fp32(None if bias is None else bias[start:end]),
            ).double()
            new_max = torch.maximum(running_max, logits.amax(dim=1))
            running_sum.mul_((running_max - new_max).exp_()).add_(
                logits.sub_(new_max.unsqueeze(1)).exp_().sum(dim=1)
            )
            running_max = new_max
        # Release the full chunk before allocating selected-class logits.
        del logits
        normalizer = running_max.add_(running_sum.log_())
        selected_weight = _at_least_fp32(weight.index_select(0, selected))
        selected_bias = None if bias is None else _at_least_fp32(bias.index_select(0, selected))
        log_probs = _chunk_logits(flat, selected_weight, selected_bias)
        # Subtract in float64, then round once to the DP dtype.
        log_probs = log_probs.double().sub_(normalizer.unsqueeze(1)).to(dp_dtype)
        ctx.save_for_backward(hidden, weight, bias, selected, normalizer.to(dp_dtype), log_probs)
        ctx.chunk_size = chunk_size
        return log_probs.view(*hidden.shape[:-1], selected.numel())

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_log_probs: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        hidden, weight, bias, selected, normalizer, saved_log_probs = ctx.saved_tensors
        # Backward can run outside the caller's forward autocast context.
        with torch.autocast(hidden.device.type, enabled=False):
            chunk_size = ctx.chunk_size
            hidden_dim = hidden.shape[-1]
            vocab_size = weight.shape[0]
            need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
            need_bias = need_bias and bias is not None
            flat = _at_least_fp32(hidden.reshape(-1, hidden_dim))
            grad = _at_least_fp32(grad_log_probs.reshape(-1, selected.numel()))
            # Preserve upstream scaling and zero gradients at omitted frames.
            negative_sum = (-grad.sum(dim=1)).unsqueeze(1)
            grad_hidden = torch.zeros_like(flat) if need_hidden else None
            grad_weight = torch.empty_like(weight) if need_weight else None
            grad_bias = torch.empty_like(bias) if need_bias else None
            cast_weight = need_weight and weight.dtype not in (torch.float32, torch.float64)
            cast_bias = need_bias and bias.dtype not in (torch.float32, torch.float64)
            # Selected IDs need not be sorted when the original blank ID is nonzero.
            chunk_ids = torch.div(selected, chunk_size, rounding_mode="floor")
            for chunk_index, start in enumerate(range(0, vocab_size, chunk_size)):
                end = min(start + chunk_size, vocab_size)
                weight_chunk = _at_least_fp32(weight[start:end])
                grad_logits = _chunk_logits(
                    flat, weight_chunk, _at_least_fp32(None if bias is None else bias[start:end])
                )
                grad_logits.sub_(normalizer.unsqueeze(1)).exp_().mul_(negative_sum)
                positions = (chunk_ids == chunk_index).nonzero().flatten()
                if positions.numel():
                    local_ids = selected.index_select(0, positions) - start
                    # Reuse forward probabilities and combine cancelling terms before reductions.
                    grad_logits.index_copy_(
                        1,
                        local_ids,
                        saved_log_probs.index_select(1, positions).exp().mul_(negative_sum),
                    )
                    grad_logits.index_add_(1, local_ids, grad.index_select(1, positions))
                if need_hidden:
                    grad_hidden.addmm_(grad_logits, weight_chunk)
                if need_weight:
                    if cast_weight:
                        grad_weight[start:end].copy_(torch.mm(grad_logits.t(), flat))
                    else:
                        torch.mm(grad_logits.t(), flat, out=grad_weight[start:end])
                if need_bias:
                    if cast_bias:
                        grad_bias[start:end].copy_(grad_logits.sum(dim=0))
                    else:
                        torch.sum(grad_logits, dim=0, out=grad_bias[start:end])
            return (
                grad_hidden.view_as(hidden).to(hidden.dtype) if need_hidden else None,
                grad_weight,
                grad_bias,
                None,
                None,
            )


def pruned_ctc_loss(
    encoder_out: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    y: k2.RaggedTensor,
    encoder_out_lens: Tensor,
    *,
    chunk_size: int = 4096,
    output_beam: float = 100.0,
    max_states: int = DEFAULT_MAX_STATES,
    blank_id: int = 0,
) -> Tensor:
    """Compute summed CTC loss with reduced vocabulary and pruned alignments.

    Args:
        encoder_out: Encoder states of shape (N, T, D), in float32 or bfloat16.
        weight: Projection weights of shape (V, D), in float32 or bfloat16.
        bias: Projection bias of shape (V,), in float32 or bfloat16, or None.
        y: Two-axis k2.RaggedTensor of target IDs, excluding blank_id.
        encoder_out_lens: Int32/int64 frame counts of shape (N,), each in [1, T].
        chunk_size: Positive number of vocabulary columns per projection chunk.
        output_beam: Positive finite log-score beam, representable in float32.
        max_states: Positive lattice state limit. k2 can tighten the effective beam.
        blank_id: Blank ID in the original vocabulary, remapped to column zero.

    States, projection parameters, and targets must share a CPU or CUDA device.
    Floating-point inputs must be finite. Lengths may remain on CPU.
    Disable autocast when calling this function;
    gradients retain each floating-point input's dtype. Vocabulary reduction is
    exact, while finite-beam alignment pruning is approximate. k2's lattice
    limits can further restrict the retained alignments.

    Returns:
        A float32 scalar summed over utterances. Nonfinite per-utterance losses
        are excluded with a logged warning. Impossible alignments contribute
        zero gradients when the floating-point inputs are finite.
        Only first-order differentiation is supported.

    Raises:
        ImportError: A compatible k2 installation is unavailable.
        TypeError: An input has an unsupported type or dtype.
        ValueError: A shape, device, index, or configuration value is invalid.
        RuntimeError: Autocast is enabled on the input device.
    """
    chunk_size = _positive_integer("chunk_size", chunk_size)
    max_states = _positive_integer("max_states", max_states)
    if max_states > _INT32_MAX:
        raise ValueError("max_states must fit in a signed int32.")
    if isinstance(output_beam, bool) or not isinstance(output_beam, Real):
        raise TypeError("output_beam must be a real number.")
    output_beam = float(output_beam)
    beam_limits = torch.finfo(torch.float32)
    if not math.isfinite(output_beam) or not (
        beam_limits.tiny * beam_limits.eps <= output_beam <= beam_limits.max
    ):
        raise ValueError("output_beam must be positive, finite, and representable in float32.")
    backend = _load_k2()
    lengths_cpu = _validate_inputs(
        encoder_out, weight, bias, y, encoder_out_lens, blank_id, backend
    )
    chunk_size = min(chunk_size, weight.shape[0])
    blank_id = int(blank_id)
    device = encoder_out.device
    batch_size, max_frames, hidden_dim = encoder_out.shape
    targets = y.values.to(torch.int32)
    distinct = torch.unique(targets, sorted=True)
    selected = torch.cat((targets.new_full((1,), blank_id), distinct))
    if selected.numel() > MAX_REDUCED_VOCAB:
        if batch_size == 1:
            raise ValueError(
                f"One utterance needs {selected.numel()} reduced classes; "
                f"k2 supports at most {MAX_REDUCED_VOCAB}. Shorten its transcript."
            )
        midpoint = batch_size // 2
        _LOGGER.warning(
            "Reduced vocabulary has %d classes; splitting %d utterances to fit k2's limit.",
            selected.numel(),
            batch_size,
        )
        return sum(
            pruned_ctc_loss(
                encoder_out=encoder_out[start:end],
                weight=weight,
                bias=bias,
                y=y[start:end],
                encoder_out_lens=encoder_out_lens[start:end],
                chunk_size=chunk_size,
                output_beam=output_beam,
                max_states=max_states,
                blank_id=blank_id,
            )
            for start, end in ((0, midpoint), (midpoint, batch_size))
        )
    remapped = (torch.searchsorted(distinct, targets) + 1).to(torch.int32)
    num_valid_frames = int(lengths_cpu.sum())
    packed = num_valid_frames < batch_size * max_frames
    if packed:
        lengths_device = lengths_cpu.to(device=device)
        starts_device = torch.cumsum(lengths_device, dim=0) - lengths_device
        utterance_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device), lengths_device, output_size=num_valid_frames
        )
        frame_ids = torch.arange(num_valid_frames, device=device) - starts_device[utterance_ids]
        hidden = encoder_out.reshape(-1, hidden_dim).index_select(
            0, utterance_ids * max_frames + frame_ids
        )
        starts_cpu = torch.cumsum(lengths_cpu, dim=0) - lengths_cpu
        sequence_ids_cpu = torch.zeros(batch_size, dtype=torch.int64, device="cpu")
    else:
        hidden = encoder_out
        starts_cpu = torch.zeros(batch_size, dtype=torch.int64, device="cpu")
        sequence_ids_cpu = torch.arange(batch_size, dtype=torch.int64, device="cpu")
    log_probs = _FusedReducedLogSoftmax.apply(hidden, weight, bias, selected.long(), chunk_size)
    if packed:
        log_probs = log_probs.unsqueeze(0)
    # k2 creates CPU supervision indices internally, regardless of the caller's default device.
    with torch.device("cpu"):
        decoding_graph = backend.ctc_graph(backend.RaggedTensor(y.shape, remapped), modified=False)
        # A contiguous length vector avoids stride issues for a single utterance.
        # Apply the same stable permutation to segments and graphs.
        order = torch.argsort(lengths_cpu, descending=True, stable=True)
        segments = torch.stack(
            (sequence_ids_cpu[order], starts_cpu[order], lengths_cpu[order]), dim=1
        ).to(torch.int32)
        decoding_graph = backend.index_fsa(
            decoding_graph, order.to(device=device, dtype=torch.int32)
        )
        dense_fsa = backend.DenseFsaVec(log_probs, segments)
        lattice = backend.intersect_dense(
            a_fsas=decoding_graph,
            b_fsas=dense_fsa,
            output_beam=output_beam,
            max_states=max_states,
        )
        losses = -lattice.get_tot_scores(log_semiring=True, use_double_scores=True).float()
    global _warned_near_cap
    num_states = lattice.arcs.tot_size(1)
    if num_states > 0.5 * max_states and not _warned_near_cap:
        _warned_near_cap = True
        _LOGGER.warning(
            "Output lattice uses %d states (max_states=%d). k2 may tighten the beam "
            "when lattice limits are reached; this warning is a heuristic.",
            num_states,
            max_states,
        )
    loss = losses.sum()
    if bool(torch.isfinite(loss)):
        return loss
    finite = torch.isfinite(losses)
    invalid_indices = order[~finite.cpu()].tolist()
    _LOGGER.warning(
        "Dropping %d utterances with nonfinite CTC loss; batch indices: %s.",
        len(invalid_indices),
        invalid_indices,
    )
    return torch.where(finite, losses, losses.new_zeros(())).sum()
