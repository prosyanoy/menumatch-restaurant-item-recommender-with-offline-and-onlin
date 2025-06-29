from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..context import DAYPARTS
from ..features import FeatureSet
from .base import Batch, Catalog, Recommender, candidate_mask

DEFAULTS = {"dim": 32, "hidden": 64, "epochs": 30, "batch_size": 512, "lr": 3e-3,
            "weight_decay": 1e-5, "dropout": 0.1, "patience": 5, "temperature": 0.1, "seed": 0}


class _TwoTower(nn.Module):
    def __init__(self, catalog: Catalog, dim: int, hidden: int, dropout: float, temperature: float) -> None:
        super().__init__()
        n, r = catalog.n_items, catalog.n_restaurants
        self.n_items = n
        # buffers describing items for the item tower
        self.register_buffer("item_restaurant", torch.as_tensor(np.array(catalog.restaurant), dtype=torch.long))
        self.register_buffer("item_category", torch.as_tensor(np.array(catalog.category), dtype=torch.long))
        price = np.log1p(catalog.price)
        self.register_buffer("item_price", torch.as_tensor((price - price.mean()) / (price.std() + 1e-6), dtype=torch.float32)[:, None])
        # query tower: cart (mean of item embeddings; index n = padding), restaurant, daypart, weekend
        self.cart_emb = nn.Embedding(n + 1, dim, padding_idx=n)
        self.q_rest = nn.Embedding(r, 8)
        self.q_daypart = nn.Embedding(len(DAYPARTS), 8)
        self.q_weekend = nn.Embedding(2, 4)
        self.query_mlp = nn.Sequential(nn.Linear(dim + 20, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        # item tower: id, category, restaurant, price
        self.item_emb = nn.Embedding(n, dim)
        self.i_cat = nn.Embedding(int(catalog.category.max()) + 1, 8)
        self.i_rest = nn.Embedding(r, 8)
        self.item_mlp = nn.Sequential(nn.Linear(dim + 17, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.register_buffer("scale", torch.tensor(1.0 / temperature))

    def encode_query(self, rest, cart_pad, daypart, weekend):
        mask = (cart_pad != self.n_items).float().unsqueeze(-1)
        cart = (self.cart_emb(cart_pad) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([cart, self.q_rest(rest), self.q_daypart(daypart), self.q_weekend(weekend)], dim=-1)
        return F.normalize(self.query_mlp(x), dim=-1)

    def encode_items(self):
        x = torch.cat([self.item_emb.weight, self.i_cat(self.item_category), self.i_rest(self.item_restaurant), self.item_price], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def forward(self, rest, cart_pad, daypart, weekend):
        return self.encode_query(rest, cart_pad, daypart, weekend) @ self.encode_items().T * self.scale


def _pad_carts(carts: list[np.ndarray], pad: int) -> torch.Tensor:
    width = max(1, max((len(c) for c in carts), default=1))
    out = np.full((len(carts), width), pad, dtype=np.int64)
    for i, c in enumerate(carts):
        out[i, : len(c)] = c
    return torch.as_tensor(out)


def _tensors(batch: Batch, n_items: int):
    return (torch.as_tensor(batch.restaurant), _pad_carts(batch.carts, n_items),
            torch.as_tensor(batch.daypart), torch.as_tensor(batch.weekend))


class TwoTowerRecommender(Recommender):
    """PyTorch two-tower retrieval model trained with a masked full softmax.

    Because each restaurant has only a few dozen add-ons, the softmax is computed
    over *all* valid candidates (same restaurant, add-on, not in cart) rather than
    sampled negatives; this matches the inference-time candidate set exactly.
    """

    kind = "two_tower"

    def __init__(self, catalog: Catalog, **hparams) -> None:
        super().__init__(catalog)
        self.hp = {**DEFAULTS, **hparams}
        self.net: _TwoTower | None = None
        self.history: list[dict] = []

    def _build(self) -> _TwoTower:
        hp = self.hp
        torch.manual_seed(hp["seed"])
        return _TwoTower(self.catalog, hp["dim"], hp["hidden"], hp["dropout"], hp["temperature"])

    def fit(self, fs: FeatureSet, val=None) -> "TwoTowerRecommender":
        from ..evaluation import evaluate  # local import avoids a cycle

        hp = self.hp
        val = fs.val if val is None else val
        self.net = net = self._build()
        opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        train = Batch.from_examples(fs.train)
        targets = torch.as_tensor(fs.train.target_item.to_numpy(copy=True), dtype=torch.long)
        masks = torch.as_tensor(candidate_mask(self.catalog, train))
        tens = _tensors(train, self.catalog.n_items)
        gen = torch.Generator().manual_seed(hp["seed"])
        best, best_state, stale = -1.0, None, 0
        self.history = []
        for epoch in range(hp["epochs"]):
            net.train()
            perm = torch.randperm(len(targets), generator=gen)
            total = 0.0
            for s in range(0, len(perm), hp["batch_size"]):
                idx = perm[s : s + hp["batch_size"]]
                cart = tens[1][idx]
                logits = net(tens[0][idx], cart, tens[2][idx], tens[3][idx]).masked_fill(~masks[idx], -1e9)
                loss = F.cross_entropy(logits, targets[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            score = evaluate(self, val, ks=(5,), n_boot=0)["ndcg@5"] if len(val) else -total
            self.history.append({"epoch": epoch, "train_loss": total / len(perm), "val_ndcg@5": score})
            if score > best:
                best, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= hp["patience"]:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return self

    @torch.no_grad()
    def score(self, batch: Batch) -> np.ndarray:
        assert self.net is not None, "model is not fitted"
        self.net.eval()
        return self.net(*_tensors(batch, self.catalog.n_items)).numpy()

    def get_state(self):
        arrays = {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}
        return arrays, {"hparams": self.hp, "history": self.history}

    @classmethod
    def from_state(cls, arrays, meta, catalog):
        model = cls(catalog, **meta["hparams"])
        model.net = model._build()
        model.net.load_state_dict({k: torch.as_tensor(np.array(v)) for k, v in arrays.items()})
        model.net.eval()
        model.history = meta.get("history", [])
        return model
