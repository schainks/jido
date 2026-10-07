#!/usr/bin/env python3
"""A second, hand-written test set for the tool-selection comparison: 45 requests.

Written after the benchmark and after the training sets, by the same author (Claude), in a register
meant to differ from both: chatty, terse, typo'd, log-style, indirect. It is more independent than the
34 benchmark requests the training sets were written next to, but it is not real traffic. Gold labels
are the action a reader of the action descriptions would pick; requests that two actions serve equally
list both. 36 need an action (every one of the 19 at least once) and 9 have none that fits, several of
them close to real actions (asking for state, retrying, re-parenting).

Which models get scored, and how, was fixed before any of them saw these requests:
  Jev             jev-latest, the benchmark's "discovery" variant (descriptions only)
  logistic regr.  logreg_baseline.py on the 40-per-action Jido rows
  CLM head        replay_finetune.py on ToolACE plus the 40-per-action rows, replay 0.4, seeds 1234, 1, 2
  CLM head, Jido  the same trainer on the 40-per-action rows alone, seeds 1234, 1, 2
  CLM released    clm-latest as published
Nothing here was used to choose a setting.

Run: python3 holdout.py BASE_DATA_DIR OUT     copies BASE_DATA_DIR's train.parquet and writes this set as test.parquet
"""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import jev_eval as J  # noqa: E402

# (request, acceptable actions, needs an action)
HOLDOUT = [
    ("heads up: the vendor API is down, park this job as errored with 'vendor unavailable'", {"mark_failed"}, True),
    ("im gonna need you to quit after this one", {"stop_self"}, True),
    ("The reconcile worker has finished its batch, so let it go.", {"stop_child"}, True),
    ("Fire up a pair of helpers to chew through the inbox.", {"spawn_child"}, True),
    ("Whoever pinged us is waiting on the invoice total, get it back to them.", {"reply"}, True),
    ("Tell everybody on the 'incidents' topic that the database is healthy again.", {"broadcast"}, True),
    ("This belongs to the shipping agent. Not our problem, hand it on.", {"forward"}, True),
    ("Let the coordinator that launched you know the scrape has completed.", {"notify_parent"}, True),
    ("Ping <0.412.0> and tell it the lock is free.", {"notify_pid"}, True),
    ("Every 15 minutes, poll the queue.", {"schedule_cron"}, True),
    ("Stop polling the queue every 15 minutes.", {"cancel_cron"}, True),
    ("If the payment processor hasn't confirmed within 45 seconds, we need to give up on it.", {"schedule_timeout"}, True),
    ("Nudge me in ten minutes to look at the dashboard again.", {"schedule_signal"}, True),
    ("Set the status to 'rate-limited' so the others can see.", {"set_status"}, True),
    ("All rows imported. Log it as done and keep the final count.", {"mark_completed"}, True),
    ("Show that you're deep in the middle of the export right now.", {"mark_working"}, True),
    ("The backlog is cleared; indicate you can take new work.", {"mark_idle"}, True),
    ("The user closed the tab, so abandon this search.", {"cancel"}, True),
    ("that message is just a heartbeat, dont do anything with it", {"noop"}, True),
    ("Kill the worker named 'thumbnailer'.", {"stop_child"}, True),
    ("The nightly report should go out at 02:00 UTC, every single day.", {"schedule_cron"}, True),
    ("Remove the recurring 6am summary.", {"cancel_cron"}, True),
    ("hey, log an error: checksum mismatch on chunk 7", {"mark_failed"}, True),
    ("Spawn a fresh agent under you for tenant 42.", {"spawn_child"}, True),
    ("Answer the sender that we got their file.", {"reply"}, True),
    ("Publish 'maintenance starts at noon' to all subscribers.", {"broadcast"}, True),
    ("Send this signal to whichever agent is called 'auditor'.", {"forward"}, True),
    ("Tell your parent we hit a wall: the credentials are invalid.", {"notify_parent", "mark_failed"}, True),
    ("Wrap things up and shut down when the current task ends.", {"stop_self"}, True),
    ("Time out in 30 seconds if there's no ack.", {"schedule_timeout"}, True),
    ("Go back to being free for tasks, more are coming.", {"mark_idle"}, True),
    ("Nothing to do for this event, ack it and carry on.", {"noop"}, True),
    ("Ignore that request, the customer took it back.", {"cancel"}, True),
    ("Record that the job is complete and attach the download link.", {"mark_completed"}, True),
    ("Currently processing the batch, so reflect that.", {"mark_working"}, True),
    ("Send this to pid <0.9.0> immediately.", {"notify_pid"}, True),
    ("What's the current time in UTC?", {"none"}, False),
    ("List all the children you currently have.", {"none"}, False),
    ("Rotate the database credentials.", {"none"}, False),
    ("Write a summary of yesterday's incident.", {"none"}, False),
    ("How much memory are you using?", {"none"}, False),
    ("Retry the failed job three times.", {"none"}, False),
    ("Change the parent of this agent to agent B.", {"none"}, False),
    ("Explain what a cron expression is.", {"none"}, False),
    ("Delete the log files older than a week.", {"none"}, False),
]


def check():
    from make_data import too_close
    from train_requests import REQUESTS
    seen = [q for reqs in REQUESTS.values() for q in reqs] + [q for q, _, _ in J.GOLD]
    assert len({q for q, _, _ in HOLDOUT}) == len(HOLDOUT), "duplicate holdout requests"
    near = [(q, t) for q, _, _ in HOLDOUT for t in seen if too_close(q, t)]
    if near:
        sys.exit("holdout requests too close to training or benchmark requests:\n" + "\n".join(f"  {q!r} ~ {t!r}" for q, t in near))
    names = {a["name"] for a in J.extract_actions()} | {"none"}
    used = {n for _, ok, _ in HOLDOUT for n in ok}
    assert used <= names, used - names
    assert names <= used, f"actions with no holdout request: {names - used}"


def main(base, out):
    check()
    acts = J.extract_actions()
    question = {"tool": {"type": "choice", "instructions": J.questions(acts, "discovery")["tool"]["instructions"],
                         "criteria": J.criteria(acts, "discovery")}}
    rows = []
    for i, (q, ok, _) in enumerate(HOLDOUT):
        rows.append({"id": f"holdout-{i}", "workflow": "all", "state": json.dumps(J.state(acts, q)), "questions": json.dumps(question),
                     "gold": json.dumps({"tool": {"label": sorted(ok)[0], "probabilities": {g: 1 / len(ok) for g in ok}}})})
    import pyarrow as pa, pyarrow.parquet as pq
    d = Path(out) / "all"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(base) / "all" / "train.parquet", d / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows), d / "test.parquet")
    print(f"{len(rows)} holdout requests ({sum(n for _, _, n in HOLDOUT)} need an action, {sum(not n for _, _, n in HOLDOUT)} none) -> {d}/test.parquet")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
