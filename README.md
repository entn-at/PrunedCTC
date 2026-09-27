# Pruned CTC

Memory-efficient CTC training for large vocabularies, implemented in PyTorch and
[k2](https://github.com/k2-fsa/k2).

Pruned CTC computes the CTC loss directly from encoder states and a linear
projection, avoiding a full `(batch, time, vocabulary)` logit tensor. It combines:

- **Exact vocabulary reduction:** the CTC graph uses only the target tokens and
  blank. Probabilities are still normalized over the **full vocabulary**, and
  gradients include every vocabulary class.
- **Chunked projection and backward:** vocabulary chunks are recomputed during
  backward to reduce the memory used by the projection and loss activations.
- **Alignment pruning:** k2 prunes the alignment lattice using a configurable
  log-score beam. A finite beam makes the alignment sum approximate; vocabulary reduction
  itself does not introduce this approximation.

The projection weights, their gradients, and optimizer states remain dense.
Memory savings depend on the vocabulary, batch, sequence lengths, chunk size,
and alignment beam.

## Installation

Requires **Python 3.10+**, **PyTorch 2.4+**, and a compatible **k2** build.
First install PyTorch and k2 for your Python version, device, and CUDA version.
Follow the
[k2 installation guide](https://k2-fsa.github.io/k2/installation/index.html);
k2 wheels are tied to specific PyTorch builds. A generic `pip install k2` can
select an older PyTorch dependency and replace an existing installation.

Install Pruned CTC from this repository:

```bash
git clone https://github.com/yfyeung/PrunedCTC.git
cd PrunedCTC
python -m pip install .
```

The distribution name is `pruned-ctc`; the Python import is `pruned_ctc`.
k2 is installed separately so that Pruned CTC does not choose a binary build for
your environment.

## Quick start

```python
import k2
import torch

from pruned_ctc import pruned_ctc_loss

torch.manual_seed(0)
device = torch.device("cpu")  # Use "cuda" with a compatible CUDA-enabled k2.

encoder_out = torch.randn(2, 12, 8, device=device, requires_grad=True)
head = torch.nn.Linear(8, 6, device=device)
targets = k2.RaggedTensor([[1, 2, 3], [2, 2]]).to(device)
lengths = torch.tensor([12, 9], dtype=torch.int64, device=device)

with torch.autocast(device.type, enabled=False):
    loss = pruned_ctc_loss(
        encoder_out=encoder_out,
        weight=head.weight,
        bias=head.bias,
        y=targets,
        encoder_out_lens=lengths,
        chunk_size=4,
        output_beam=100.0,
        blank_id=0,
    )

loss.backward()
print(f"Summed CTC loss: {loss.item():.4f}")
```

## API

```python
pruned_ctc_loss(
    encoder_out,
    weight,
    bias,
    y,
    encoder_out_lens,
    chunk_size=4096,
    output_beam=100.0,
    max_states=100_000_000,
    blank_id=0,
)
```

| Argument | Expected value |
| --- | --- |
| `encoder_out` | Encoder states of shape `(N, T, D)`. |
| `weight` | Linear projection weights of shape `(V, D)`. |
| `bias` | Projection bias of shape `(V,)`, or `None`. |
| `y` | Two-axis `k2.RaggedTensor`: one sequence of target IDs per utterance, with no blank tokens. |
| `encoder_out_lens` | Int32/int64 frame counts of shape `(N,)`, each between `1` and `T`. |
| `chunk_size` | Number of vocabulary columns processed per chunk. Smaller chunks reduce temporary projection memory and can cost speed. |
| `output_beam` | Positive, finite log-score beam used for alignment pruning. Larger beams retain more alignments and use more memory. |
| `max_states` | Lattice state limit. k2 may narrow the effective beam to stay within this limit. |
| `blank_id` | Blank token ID in the original vocabulary; it is remapped internally to column zero for k2. |

The return value is a scalar `float32` **sum** over utterances. Nonfinite
per-utterance losses are excluded with a warning, including losses from
impossible target alignments. With finite inputs, impossible alignments
contribute zero loss and zero gradients. Apply any desired loss normalization
explicitly.

### Precision and device requirements

- `encoder_out`, `weight`, and `bias` may each use `float32` or `bfloat16`.
  Their values must be finite. Gradients retain the corresponding input dtypes.
  `float16` is unsupported.
- Disable autocast around the loss call, even when the encoder runs under mixed
  precision. The loss manages its own numerical precision.
- Put the encoder states, projection parameters, and targets on the same CPU or
  CUDA device. Frame counts may also stay on the CPU.
- Target IDs must be valid vocabulary indices and must exclude `blank_id`.
- Only first-order gradients are supported; double backward is unsupported.

## License

[MIT](https://github.com/yfyeung/PrunedCTC/blob/main/LICENSE). Copyright 2026 Shanghai Jiao Tong University.
Author: Yifan Yang.
