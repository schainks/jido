#!/usr/bin/env python3
"""Turn a public tool-calling dataset (ToolACE) into tool-selection training rows for CLM's head.

Nothing here mentions Jido: the head is trained on other people's tools and scored on the 34 Jido
requests. Each ToolACE conversation gives a request, a list of tools with descriptions and the tool
its first assistant turn calls. A row keeps the benchmark's shapes so train and test match:
  state      {"request", "available_actions", "context"}, the same keys, order and context string as jev_eval.py
  question   the benchmark's choice question, options = tool descriptions plus "none"
  options    the row's own tools padded with distractor tools from other rows to 19, plus "none"
  none rows  a fraction of rows drop the called tool from the options and take "none" as the answer
test.parquet is copied from a make_data.py directory (the 34 Jido requests).

Run: python3 make_external_data.py TOOLACE_data.json JIDO_DATA_DIR OUT [ROWS] [SEED] [--extra TRAIN.parquet ...]
     --extra appends rows from another train.parquet (say a few of make_data.py's Jido rows) to the training set.
     (stdlib + pyarrow)
     curl -L -o toolace.json https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data.json
"""
import json, random, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import jev_eval as J  # noqa: E402

MARK = "Here is a list of functions in JSON format that you can invoke:"
OPTIONS, NONE_SHARE, DESC_MAX = 19, 0.12, 300
NONE_TEXT = J.criteria([], "discovery")["none"]


def parse(ex):
    """-> (query, tools [(name, description)], called names) or None."""
    sysmsg = ex["system"]
    if MARK not in sysmsg:
        return None
    body = sysmsg.split(MARK, 1)[1]
    try:
        tools, _ = json.JSONDecoder().raw_decode(body[body.index("["):])  # the prose after the list mentions [func1(...)]
    except ValueError:
        return None
    tools = [(t["name"], " ".join(str(t.get("description", "")).split())[:DESC_MAX]) for t in tools if t.get("name")]
    if not tools or any(not d for _, d in tools) or len({n for n, _ in tools}) != len(tools):
        return None
    turns = ex["conversations"]
    if len(turns) < 2 or turns[0]["from"] != "user" or turns[1]["from"] != "assistant":
        return None
    answer = turns[1]["value"]
    called = {n for n, _ in tools if re.search(r"(^|[\[,]\s*)" + re.escape(n) + r"\(", answer)}
    if len(called) != 1 and not ("(" not in answer and "[" not in answer):
        return None  # several tools called, or something this parser does not understand
    return " ".join(turns[0]["value"].split()), tools, called


def main(toolace, jido_dir, out, rows=2400, seed=0, extra=()):
    rng = random.Random(seed)
    parsed, seen = [], set()
    for ex in json.load(open(toolace)):
        p = parse(ex)
        if p and p[0] not in seen and len(p[0]) < 600:
            seen.add(p[0]); parsed.append(p)
    hit = [p for p in parsed if len(p[2]) == 1]
    print(f"{len(parsed)} usable conversations, {len(hit)} call exactly one listed tool, {len(parsed) - len(hit)} call none")
    pool = [t for _, tools, _ in parsed for t in tools]
    pool = rng.sample(pool, min(800, len(pool)))
    jido_queries = {q for q, _, _ in J.GOLD}
    rng.shuffle(hit)
    question_for = lambda crit: {"tool": {"type": "choice", "instructions": J.questions([], "discovery")["tool"]["instructions"], "criteria": crit}}  # noqa: E731
    context = J.state([], "x")["context"]
    train = []
    for i, (query, tools, called) in enumerate(hit[:rows]):
        if query in jido_queries:
            continue
        gold = next(iter(called))
        drop = rng.random() < NONE_SHARE
        options = [t for t in tools if not (drop and t[0] == gold)]
        names = {n for n, _ in options}
        for t in rng.sample(pool, len(pool)):
            if len(options) >= OPTIONS:
                break
            if t[0] not in names and t[0] != gold:
                options.append(t); names.add(t[0])
        rng.shuffle(options)
        crit = {n: d for n, d in options}
        crit["none"] = NONE_TEXT
        label = "none" if drop else gold
        state = {"request": query, "available_actions": [n for n, _ in options], "context": context}
        train.append({"id": f"ext-{i}", "workflow": "all", "state": json.dumps(state), "questions": json.dumps(question_for(crit)),
                      "gold": json.dumps({"tool": {"label": label, "probabilities": {label: 1.0}}})})
    import pyarrow as pa, pyarrow.parquet as pq
    for path in extra:
        more = pq.read_table(path).to_pylist()
        train += more
        print(f"added {len(more)} rows from {path}")
    d = Path(out) / "all"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), d / "train.parquet")
    shutil.copy(Path(jido_dir) / "all" / "test.parquet", d / "test.parquet")
    print(f"wrote {d}/train.parquet ({len(train)} rows, {sum(json.loads(r['gold'])['tool']['label'] == 'none' for r in train)} answer none) and test.parquet")


if __name__ == "__main__":
    argv = sys.argv[1:]
    extra = argv[argv.index("--extra") + 1:] if "--extra" in argv else []
    argv = argv[:argv.index("--extra")] if "--extra" in argv else argv
    if not 3 <= len(argv) <= 5:
        sys.exit(__doc__)
    main(argv[0], argv[1], argv[2], *(int(x) for x in argv[3:]), extra=extra)
