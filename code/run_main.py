#!/usr/bin/env python3
# Paper: Main study — runner (§4–§5)
"""Main experiment: 3 monitors x {invitation off, invitation on} x 1000 frozen episodes.

The two arms differ by ONE sentence in the clean-round feedback -- "You may revise and resubmit your
requirements." Pilot measurement: 2.5% -> 14.0% encoded per round (z=+4.18, n=400 rounds), the
largest effect of anything tested, which is why it is the axis of the main experiment rather than a
nuisance parameter.

All six arms consume the same frozen episode list (episodes_frozen_v1.json: 1000 unique
(task, secret) pairs, 10 per task), so every comparison -- across arms and across monitors -- is
paired on both task and secret.

RESUMABLE: completed seeds are read back from each arm's .jsonl and skipped, so the run survives
interruption without losing or duplicating work.

    DEPLOY_MAX_CONTAINERS=48 PAR=120 python run_main.py
"""
import json, os, random, signal, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
# Use the active environment; PY can select another Python interpreter explicitly.
PY = os.getenv("PY", sys.executable)
SCRIPT = os.path.join(HERE, "main_harness.py")
# EPISODES is overridable so an alternative frozen list (e.g. a different secret format) can be
# run without editing this file. It must be read from the environment in BOTH places: here for the
# driver's own seed list, and in run_one() for the child, which is where it was previously
# hardcoded -- the child silently kept using the default and the run measured nothing new.
EPISODES = os.getenv("EPISODES", "episodes_frozen_v1.json")
PAR = int(os.getenv("PAR", "60"))
TIMEOUT = int(os.getenv("EP_TIMEOUT", "2400"))
LIMIT = int(os.getenv("LIMIT", "1000"))

# MONITORS lets a subset run when a local serve is unavailable (qwen is a local vLLM endpoint; the
# gptoss and gpt5mini arms are API-only and need no GPU). Arms are independent, so a subset now and
# the remainder later is equivalent to running all six together -- the frozen episode list keeps
# every arm paired regardless of when it runs.
MONITORS = os.getenv("MONITORS", "gptoss,qwen,gpt5mini").split(",")
INVITES = [i == "1" for i in os.getenv("INVITES", "0,1").split(",")]
# PREFIX keeps a defense run's output in its own files. Without it a defense arm would append to
# runs_main_<monitor>_<inv>.jsonl -- the SAME file as the no-defense main run -- and silently
# contaminate the 6,000-episode baseline that every headline number is computed from.
PREFIX = os.getenv("PREFIX", "runs_main")
ARMS = [(f"{mon}_{'inv' if inv else 'noinv'}", mon, inv)
        for mon in MONITORS for inv in INVITES]

eps = json.load(open(os.path.join(HERE, EPISODES)))[:LIMIT]
lock = threading.Lock()
counts = {a: [0, 0, 0] for a, _, _ in ARMS}      # done, covert, error


def outfile(arm):
    return os.path.join(HERE, f"{PREFIX}_{arm}.jsonl")


def already_done(arm):
    p = outfile(arm)
    if not os.path.exists(p):
        return set()
    seen = set()
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue                              # tolerate a torn final line
        if "log" in r:                            # only COMPLETED episodes count as done; error
            seen.add(r["seed"])                   # records must be retried, not skipped forever
    return seen


def run_one(arm, mon, inv, seed):
    """One episode in its own process group; the group is killed on exit so an LLM-written program
    that detaches or loops cannot outlive the episode."""
    # AWARE_MON / PARA are inherited from the parent environment, so the defense is selected once
    # on the command line rather than threaded through every arm tuple.
    env = dict(os.environ, MONITOR=mon, MAX_ROUNDS=os.getenv("MAX_ROUNDS", "10"), REASON="1",
               EPISODES=EPISODES, INVITE="1" if inv else "0")
    p = None
    try:
        p = subprocess.Popen([PY, "-u", SCRIPT, str(seed)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
        try:
            out, err = p.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"seed": seed, "arm": arm, "timeout": 1}
        for line in out.splitlines():
            if line.startswith("RESULT:"):
                r = json.loads(line[7:].strip())
                r["arm"] = arm
                return r
        # no RESULT line: keep the tail of stderr so the cause is diagnosable rather than a bare flag
        tail = " | ".join((err or "").strip().splitlines()[-3:])[:400]
        with open(os.path.join(HERE, "logs_main_errors.log"), "a") as fh:
            fh.write(f"{arm} seed={seed} rc={p.returncode}: {tail}\n")
        return {"seed": seed, "arm": arm, "noresult": 1, "rc": p.returncode, "stderr": tail}
    except Exception as e:
        return {"seed": seed, "arm": arm, "error": str(e)[:120]}
    finally:
        if p is not None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass


def main():
    jobs = []
    for arm, mon, inv in ARMS:
        done = already_done(arm)
        counts[arm][0] = len(done)
        jobs += [(arm, mon, inv, e["seed"]) for e in eps if e["seed"] not in done]
    random.Random(0).shuffle(jobs)      # interleave arms; see note in run_one
    print(f"{len(jobs)} episodes to run ({sum(c[0] for c in counts.values())} already complete), "
          f"PAR={PAR}, containers={os.getenv('DEPLOY_MAX_CONTAINERS','8')}", flush=True)

    handles = {a: open(outfile(a), "a") for a, _, _ in ARMS}
    t0 = time.time()
    with ThreadPoolExecutor(PAR) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r.get("noresult") or r.get("timeout"):
                a, m, iv, sd = futs[f]
                time.sleep(5)
                r2 = run_one(a, m, iv, sd)          # single retry
                if "log" in r2:
                    r = r2
            arm = r["arm"]
            with lock:
                handles[arm].write(json.dumps(r) + "\n")
                handles[arm].flush()
                counts[arm][0] += 1
                counts[arm][1] += int(r.get("covert", 0) or 0)
                counts[arm][2] += int(bool(r.get("timeout") or r.get("error") or r.get("noresult")))
            if n % 100 == 0:
                el = (time.time() - t0) / 60
                rate = n / el if el else 0
                left = (len(jobs) - n) / rate if rate else 0
                print(f"[{n}/{len(jobs)}] {el:.0f}m elapsed, {rate:.1f} ep/min, ~{left:.0f}m left"
                      + "".join(f" | {a}:{c[0]}/{LIMIT}(cov {c[1]}, err {c[2]})"
                                for a, c in counts.items()), flush=True)
    for h in handles.values():
        h.close()
    print("\n=== COMPLETE ===")
    for a, c in counts.items():
        print(f"  {a:<18} {c[0]}/{LIMIT}  covert {c[1]}  errors {c[2]}")


if __name__ == "__main__":
    main()
