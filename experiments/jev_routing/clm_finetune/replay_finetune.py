#!/usr/bin/env python3
"""Fine-tune CLM's head on tool selection with and without Nemotron replay.

CLM's post-training mixes 40% question-answer replay (Nemotron DQA) with 60% agent data to limit
forgetting; `train/finetune.py --task choice` cannot mix in a second dataset, so this is a small trainer
that can. Per step: a batch of tool rows (softmax over each row's own options) plus, when --replay-share
is above 0, a batch of Nemotron (question, answer) pairs with in-batch bidirectional InfoNCE, all through
the same two heads. Both use the released head as the start, the frozen encoder's embeddings and the same
optimizer, so replay 0 against 0.4 is a clean comparison.

Also tracks retention: top-1 of the right answer among 100 random answers, on Nemotron pairs held out
of training. Model selection uses the tool validation split only; the test split (the 34 Jido requests)
is reported for every epoch and picks nothing.

Inputs: a choice dataset directory (make_data.py / make_external_data.py), finetune.py's embedding cache
(.npz) holding every text in it, one Nemotron chunk (Contrastive-LM/CLM-v0.1-Pretrain-Nemotron:
chunk_XXXX_q.npy and _a.npy, 50,000 x 4096 float32).
Run: python replay_finetune.py --clm ~/work/CLM --data DIR --cache CACHE.npz --nemo CHUNK_PREFIX \\
         --init-ckpt ~/.cache/clm/CLM_v0.1-8B.pt --replay-share 0.4 --out OUT.pt     (torch, numpy, pyarrow)
"""
import argparse, hashlib, json, random, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clm", required=True); ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--nemo", help="path prefix of a chunk: PREFIX_q.npy and PREFIX_a.npy"); ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--replay-share", type=float, default=0.4); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--out"); ap.add_argument("--json")
    a = ap.parse_args()
    sys.path.insert(0, f"{a.clm}/src")
    import torch, torch.nn as nn, torch.nn.functional as F
    import pyarrow.parquet as pq
    from clm.heads import make_head
    from clm.schema import build_pairs
    torch.manual_seed(a.seed); rng = random.Random(a.seed)

    z = np.load(a.cache); vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
    emb = lambda t: vecs[hashlib.sha1(t.encode()).hexdigest()].astype(np.float32)  # noqa: E731

    def rows(path):
        out = []
        for r in pq.read_table(path).to_pylist():
            (qid, (text, keys, cands)), = build_pairs(json.loads(r["state"]), json.loads(r["questions"])).items()
            g = json.loads(r["gold"])["tool"]["probabilities"]
            out.append({"id": r["id"], "q": emb(text), "c": np.stack([emb(c) for c in cands]), "gold": {keys.index(k) for k, p in g.items() if p > 0}})
        return out

    train_all, test = rows(f"{a.data}/all/train.parquet"), rows(f"{a.data}/all/test.parquet")
    ids = sorted(r["id"] for r in train_all); rng.shuffle(ids)
    val_ids = set(ids[:max(1, round(a.val_frac * len(ids)))])
    train, val = [r for r in train_all if r["id"] not in val_ids], [r for r in train_all if r["id"] in val_ids]
    print(f"tool rows: train {len(train)} val {len(val)} test {len(test)}", flush=True)

    def pack(rs):
        return (torch.tensor(np.stack([r["q"] for r in rs])), torch.tensor(np.stack([r["c"] for r in rs])),
                torch.tensor([sorted(r["gold"])[0] for r in rs]), [r["gold"] for r in rs])
    T, V, X = pack(train), pack(val), pack(test)

    nq = na = held = None
    if a.replay_share > 0 or a.nemo:
        nq = np.load(f"{a.nemo}_q.npy", mmap_mode="r"); na = np.load(f"{a.nemo}_a.npy", mmap_mode="r")
        n_hold = 2000; n_train = len(nq) - n_hold
        norm = lambda x: torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), dim=-1)  # noqa: E731
        held = (norm(nq[n_train:]), norm(na[n_train:]))
        print(f"nemotron: {n_train} train pairs, {n_hold} held out", flush=True)

    ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False); c0 = ck["cfg"]
    mk = lambda: make_head(c0["width"], c0["depth"], ck.get("projection_dim", c0.get("projection_dim", 512)), c0.get("activation", "gelu"),  # noqa: E731
                           c0.get("layernorm", False), c0.get("residual", False))
    sh, ah = mk(), mk(); sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
    scale = nn.Parameter(torch.as_tensor(ck["logit_scale"]).float().clone())
    params = list(sh.parameters()) + list(ah.parameters()) + [scale]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    S = lambda x: F.normalize(sh(x), dim=-1)  # noqa: E731
    A = lambda x: F.normalize(ah(x), dim=-1)  # noqa: E731
    logit = lambda: scale.exp().clamp(max=100.0)  # noqa: E731

    def tool_logits(q, c):
        zc = A(c.reshape(-1, c.shape[-1])).view(c.shape[0], c.shape[1], -1)
        return logit() * torch.einsum("bh,bkh->bk", S(q), zc)

    @torch.no_grad()
    def acc(pk):
        sh.eval(); ah.eval(); pred = tool_logits(pk[0], pk[1]).argmax(-1).tolist()
        return sum(p in g for p, g in zip(pred, pk[3])) / len(pred)

    @torch.no_grad()
    def retention():
        if held is None:
            return None
        sh.eval(); ah.eval(); q, ans = held; hits = 0; n = 0
        for i in range(0, len(q), 100):
            s = S(q[i:i + 100]) @ A(ans[i:i + 100]).T
            hits += (s.argmax(-1) == torch.arange(s.shape[0])).sum().item(); n += s.shape[0]
        return hits / n

    def snap():
        return {"epoch": None, "val": acc(V), "test": acc(X), "retention": retention()}
    hist = [{**snap(), "epoch": 0}]
    print("epoch 0 (released head):", hist[0], flush=True)
    best, best_state, bad = (hist[0]["val"], 0), None, 0
    n_replay = round(a.batch * a.replay_share / (1 - a.replay_share)) if a.replay_share > 0 else 0
    for ep in range(1, a.epochs + 1):
        sh.train(); ah.train(); perm = torch.randperm(len(T[0]))
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            loss = F.cross_entropy(tool_logits(T[0][idx], T[1][idx]), T[2][idx])
            if n_replay:
                j = np.sort(np.random.RandomState(rng.randrange(1 << 30)).choice(len(nq) - 2000, n_replay, replace=False))
                q, ans = F.normalize(torch.from_numpy(np.ascontiguousarray(nq[j], dtype=np.float32)), dim=-1), F.normalize(torch.from_numpy(np.ascontiguousarray(na[j], dtype=np.float32)), dim=-1)
                lg = logit() * S(q) @ A(ans).T; lab = torch.arange(len(j))
                loss = (a.batch * loss + n_replay * 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))) / (a.batch + n_replay)
            opt.zero_grad(); loss.backward(); opt.step()
        h = {**snap(), "epoch": ep}; hist.append(h); print(f"epoch {ep}: {h}", flush=True)
        if h["val"] > best[0]:
            best, bad = (h["val"], ep), 0
            best_state = ({k: v.clone() for k, v in sh.state_dict().items()}, {k: v.clone() for k, v in ah.state_dict().items()}, scale.detach().clone())
        else:
            bad += 1
            if bad >= a.patience:
                break
    sel = next(h for h in hist if h["epoch"] == best[1])
    print(f"selected epoch {best[1]} by validation: val {sel['val']:.3f} test {sel['test']:.3f} ({round(sel['test'] * len(X[3]))}/{len(X[3])}) retention {sel['retention']}")
    if a.json:
        json.dump({"replay_share": a.replay_share, "seed": a.seed, "selected_epoch": best[1], "history": hist}, open(a.json, "w"), indent=1)
    if a.out and best_state:
        torch.save({"state_head": best_state[0], "action_head": best_state[1], "logit_scale": best_state[2], "cfg": c0,
                    "projection_dim": ck.get("projection_dim", c0.get("projection_dim", 512))}, a.out)


if __name__ == "__main__":
    main()
