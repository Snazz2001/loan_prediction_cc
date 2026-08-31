# Attribution — official DANet (WhatAShot/DANet)

This directory vendors a **minimal subset** of the official PyTorch implementation of

> Chen, Jintai; Liao, Kuanlun; Wan, Yao; Chen, Danny Z; Wu, Jian.
> *DANets: Deep Abstract Networks for Tabular Data Classification and Regression.*
> AAAI 2022. https://arxiv.org/abs/2112.02962

- Upstream repository: https://github.com/WhatAShot/DANet
- Upstream commit: `b007c57121ec9082f6ef19ec7465d9df70767c26` (main, 2022-02-14)
- License: MIT (see `LICENSE`, copyright 2021 Ronnie Rocket)

## Files copied (unchanged except the DANet.py import path)

| File | Source |
| --- | --- |
| `DANet.py` | `model/DANet.py` — `LearnableLocality` (Entmax15 k-masks), `AbstractLayer` (ABSTLAY), `BasicBlock`, `DANet` |
| `sparsemax.py` | `model/sparsemax.py` — `Entmax15` / `Sparsemax` |
| `LICENSE` | repo root |

The only code change from upstream `DANet.py` is replacing `import model.sparsemax as sparsemax` with a relative `from . import sparsemax` so this package can live under `third_party/danet/` without the original `model/` package layout.

Not vendored (unrelated to a reproducible eval of this 20-column credit model): `abstract_model.py`, `DAN_Task.py`, `main.py`, `predict.py`, `lib/*`, `config/*`, `data/*`, `model/AcceleratedModule.py`, experiment yaml, figures.

Architecture used here: `third_party.danet.DANet.DANet` constructed as

```text
DANet(input_dim, num_classes=2, layer_num, base_outdim, k, virtual_batch_size, drop_rate)
```

`LearnableLocality` applies **Entmax15** (`k`-row sparse masks) as in the official ABSTLAY. Binary PD is `softmax(logits)[:, 1]`.
