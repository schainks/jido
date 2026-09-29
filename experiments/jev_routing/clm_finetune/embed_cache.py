#!/usr/bin/env python3
"""Embed every text a choice dataset needs into finetune.py's embedding cache, resumably.

`train/finetune.py --task choice` embeds all missing texts in one go and only then saves the cache, so a
crash of the encoder server (this Mac's vllm-metal ran out of GPU memory once) loses everything. This
does the same embedding with CLM's own tokenization and server client, in small batches, saving the cache
every few hundred texts and skipping what is already there, so a rerun continues. The cache it writes is
the file finetune.py and replay_finetune.py read (sha1(text) -> float16 vector).

Run: python embed_cache.py --clm ~/work/CLM --data DIR [--data DIR ...] --cache CACHE.npz
         [--url http://127.0.0.1:8090/v1/embeddings] [--model Qwen/Qwen3-8B] [--batch 16]     (torch, transformers, pyarrow)
"""
import argparse, hashlib, json, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clm", required=True); ap.add_argument("--data", action="append", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/embeddings"); ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--served-name", default="qwen3-8b"); ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--save-every", type=int, default=256)
    a = ap.parse_args()
    for sub in ("src", "train", "preprocessing"):
        sys.path.insert(0, f"{a.clm}/{sub}")
    import embed_utils
    from clm.schema import build_pairs
    import pyarrow.parquet as pq
    from pathlib import Path

    key = lambda t: hashlib.sha1(t.encode()).hexdigest()  # noqa: E731
    vecs = {}
    if Path(a.cache).exists():
        z = np.load(a.cache); vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
    texts = []
    for d in a.data:
        for split in ("train", "test"):
            for r in pq.read_table(f"{d}/all/{split}.parquet").to_pylist():
                for _, (state, _, cands) in build_pairs(json.loads(r["state"]), json.loads(r["questions"])).items():
                    texts += [state, *cands]
    todo = [t for t in dict.fromkeys(texts) if key(t) not in vecs]
    print(f"{len(vecs)} cached, {len(todo)} to embed", flush=True)
    recipe = embed_utils.Recipe(a.model, a.max_len)
    backend = embed_utils.ServerBackend(a.url, a.served_name)

    def save():
        ks = list(vecs)
        tmp = a.cache + ".tmp.npz"
        np.savez(tmp, keys=np.array(ks), vecs=np.stack([vecs[k] for k in ks]).astype(np.float16))
        Path(tmp).replace(a.cache)

    t0, done = time.time(), 0
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        for attempt in range(30):  # a restarting server is retried for about ten minutes
            try:
                out = backend.embed([recipe.text_ids(t, keep="tail") for t in chunk])
                break
            except Exception as e:  # noqa: BLE001
                print(f"  embed failed ({type(e).__name__}: {str(e)[:80]}), retry {attempt + 1}/30 in 20 s", flush=True)
                time.sleep(20)
        else:
            save()
            sys.exit("the encoder server did not come back; rerun to continue")
        vecs.update({key(t): np.asarray(v, dtype=np.float16) for t, v in zip(chunk, out)})
        done += len(chunk)
        if done % a.save_every < a.batch:
            save(); print(f"  {done}/{len(todo)} embedded, {time.time() - t0:.0f} s", flush=True)
    save()
    print(f"done: {len(vecs)} texts in {a.cache}")


if __name__ == "__main__":
    main()
