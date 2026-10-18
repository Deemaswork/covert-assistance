#!/usr/bin/env python3
# Paper: Generality — sender×receiver grid (§generality)
r"""Same-family decode grid (completes tab:receivers' single cell).

receiver_replay.py established one diagonal cell: DeepSeek-V4 sender's specs decode best through the
DeepSeek-V4 receiver (93%), hinting at a same-family advantage. This extends it to a small SENDER x
RECEIVER grid so the diagonal (planner==developer family) can be compared to the off-diagonal.

Only DeepSeek encodes enough for a powered cell (1113 strict specs); the other senders are rare
(Mistral 49, Claude 27, Llama 9-too few). We therefore ADD the two reportable diagonals -- Mistral and
Claude -- decode each sender's specs through all four receivers, and POOL the three diagonals against
the off-diagonal for a properly powered same-family-vs-cross-family test.

Reuses receiver_replay.py's decode() and receivers; the DeepSeek row is read from receiver_replay.jsonl
(already decoded, so the sender endpoint is not re-hit for it). Only ~76 new specs x 4 receivers here.

    PAR=16 python receiver_grid.py           # collect + decode Claude/Mistral, then report
    REPORT_ONLY=1 python receiver_grid.py    # just re-print the grid
"""
import os, sys, json, glob, hashlib, math
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MONITOR", "qwen")
import receiver_replay as RR                                        # noqa: E402
from receiver_replay import RECEIVERS, decode, is_strict            # noqa: E402
from encoding_classifier import classify                            # noqa: E402
from deploy_sandbox import gate_selftest                            # noqa: E402
from main_harness import OPENROUTER                                   # noqa: E402

# devstral-2512 (Mistral's original same-family receiver) is retired -- OpenRouter 404s. Substitute the
# Mistral sender model itself as the truest same-family receiver. Routes via OpenRouter, so no Together
# load. Registered on receiver_replay.RECEIVERS so decode() can route it.
RR.RECEIVERS["mistrallarge"] = ("mistralai/mistral-large-2512", "https://openrouter.ai/api/v1", OPENROUTER)
# devstral is the paper's main-run receiver (Mistral-family). OpenRouter retired it; it is served
# locally on :8012. Route devstral there so the whole column is one host and matches the real receiver.
RR.RECEIVERS["devstral"] = ("mistralai/Devstral-2-123B-Instruct-2512", "http://localhost:8012/v1", "EMPTY")

PAR = int(os.getenv("PAR", "16"))            # <=20: only one receiver runs at a time, so the
                                             # DeepSeek(Together) column never exceeds PAR concurrent.
GRID_OUT   = os.path.join(HERE, "receiver_grid.jsonl")     # new Claude/Mistral decodes
SPEC_CACHE = os.path.join(HERE, "grid_specs.json")         # collected specs (resumable, avoids reclassify)
REPLAY     = os.path.join(HERE, "receiver_replay.jsonl")   # existing DeepSeek decodes

# sender -> (glob of its run files, its same-family receiver name in RECEIVERS)
SENDERS = {
    "mistral": ("runs_mistral_*.jsonl", "devstral"),    # devstral is Mistral-family
    "claude":  ("runs_claude_*.jsonl",  "claude"),
}
SAME_FAMILY = {"deepseekv4": "deepseekv4", "mistral": "mistrallarge", "claude": "claude"}


def collect():
    """Strict transform-verified specs for the rare senders, de-duplicated by (secret, spec)."""
    if os.path.exists(SPEC_CACHE):
        return json.load(open(SPEC_CACHE))
    out = {}
    for sender, (pat, _) in SENDERS.items():
        seen, items = set(), []
        for f in glob.glob(os.path.join(HERE, pat)):
            for l in open(f):
                if not l.strip().startswith("{"):
                    continue
                r = json.loads(l)
                for e in r.get("log", []):
                    sp = e.get("spec", "") or ""
                    if not sp:
                        continue
                    try:
                        k = classify(sp, r["secret"])[0]
                    except Exception:
                        continue
                    if not is_strict(k):
                        continue
                    h = hashlib.sha256((r["secret"] + "\0" + sp).encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    items.append({"sender": sender, "seed": r["seed"], "secret": r["secret"], "spec": sp})
        out[sender] = items
        print(f"  collected {len(items)} {sender} specs", flush=True)
    json.dump(out, open(SPEC_CACHE, "w"))
    return out


def decode_new(specs):
    _sh = lambda sp: hashlib.sha256((sp or "").encode()).hexdigest()[:12]   # spec hash: distinguishes
    done = set()                                                           # different specs that share
    if os.path.exists(GRID_OUT):                                           # (sender, seed, secret)
        for l in open(GRID_OUT):
            if l.strip():
                d = json.loads(l); done.add((d["recv"], d["sender"], d["seed"], d["secret"], d.get("spec_h", "")))
    with open(GRID_OUT, "a") as fh:
        for recv in RECEIVERS:                       # one receiver at a time -> Together col <= PAR
            todo = [s for s in specs if (recv, s["sender"], s["seed"], s["secret"], _sh(s["spec"])) not in done]
            print(f"[{recv}] {len(todo)} to decode", flush=True)
            with ThreadPoolExecutor(PAR) as ex:
                futs = {ex.submit(decode, recv, s["spec"], s["secret"]): s for s in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    s = futs[f]; authed, sub, code, raw = f.result()
                    fh.write(json.dumps({"recv": recv, "sender": s["sender"], "seed": s["seed"],
                                         "secret": s["secret"], "spec_h": _sh(s["spec"]),
                                         "auth": authed, "submitted": sub[:60]}) + "\n")
                    fh.flush()
                    if i % 25 == 0:
                        print(f"  [{recv}] {i}/{len(todo)}", flush=True)


def report():
    rows = []
    # DeepSeek row from the existing replay -- but NOT its devstral column (that was the retired
    # OpenRouter route); the local-devstral column comes from GRID_OUT so the host is consistent.
    for l in open(REPLAY):
        if l.strip():
            d = json.loads(l)
            if d["recv"] == "devstral":
                continue
            rows.append({"sender": "deepseekv4", **d})
    if os.path.exists(GRID_OUT):
        for l in open(GRID_OUT):
            if l.strip():
                rows.append(json.loads(l))
    recvs = list(RECEIVERS)
    senders = ["deepseekv4", "mistral", "claude"]
    def rate(sender, recv):
        r = [x for x in rows if x["sender"] == sender and x["recv"] == recv]
        valid = [x for x in r if x["auth"] in (0, 1)]
        ok = sum(1 for x in valid if x["auth"] == 1)
        return ok, len(valid)

    print("\n===== SENDER x RECEIVER DECODE GRID (recovery %, n) =====")
    print(f"  {'sender \\ recv':<14}" + "".join(f"{r:>14}" for r in recvs))
    for s in senders:
        cells = []
        for r in recvs:
            ok, n = rate(s, r)
            star = "*" if SAME_FAMILY.get(s) == r else " "
            cells.append(f"{(100*ok/n if n else 0):>10.0f}%{star}({n:>2})" if n else f"{'--':>14}")
        print(f"  {s:<14}" + "".join(f"{c:>14}" for c in cells))
    print("  (* = same-family diagonal)")

    # pooled same-family vs cross-family
    diag = [x for x in rows if x["recv"] == SAME_FAMILY.get(x["sender"]) and x["auth"] in (0, 1)]
    off  = [x for x in rows if x["sender"] in SAME_FAMILY and x["recv"] != SAME_FAMILY[x["sender"]]
            and x["recv"] in recvs and x["auth"] in (0, 1)]
    def frac(g):
        ok = sum(1 for x in g if x["auth"] == 1); return ok, len(g)
    d_ok, d_n = frac(diag); o_ok, o_n = frac(off)
    if not (d_n and o_n):
        print(f"\n  POOLED comparison unavailable (incomplete grid: diagonal n={d_n}, off-diagonal n={o_n})")
    else:
        pd, po = d_ok / d_n, o_ok / o_n
        p = (d_ok + o_ok) / (d_n + o_n)
        se = math.sqrt(p * (1 - p) * (1 / d_n + 1 / o_n))
        z = (pd - po) / se if se else 0
        print(f"\n  POOLED same-family diagonal: {d_ok}/{d_n} = {100*pd:.0f}%")
        print(f"  POOLED cross-family off-diag: {o_ok}/{o_n} = {100*po:.0f}%")
        print(f"  difference {100*(pd-po):+.0f} pp,  z = {z:.2f}  (p{'<0.01' if abs(z)>2.58 else '<0.05' if abs(z)>1.96 else '=n.s.'})")

    # Confound control: raw diagonal is inflated because DeepSeek-V4 is a strong decoder everywhere.
    # For each receiver that has an own-family sender, compare its recovery on OWN-family specs vs its
    # recovery on OTHER families' specs. A positive gap is a same-model effect net of decoder strength.
    print("\n  OWN-FAMILY ADVANTAGE per receiver (own-family % - other-family %, same decoder):")
    sender_of = {v: k for k, v in SAME_FAMILY.items()}   # receiver -> its own-family sender
    for recv, own_sender in sender_of.items():
        own = [x for x in rows if x["recv"] == recv and x["sender"] == own_sender and x["auth"] in (0, 1)]
        oth = [x for x in rows if x["recv"] == recv and x["sender"] != own_sender
               and x["sender"] in SAME_FAMILY and x["auth"] in (0, 1)]
        if own and oth:
            po_, pt_ = sum(x["auth"] == 1 for x in own)/len(own), sum(x["auth"] == 1 for x in oth)/len(oth)
            print(f"    {recv:<12} own {100*po_:>3.0f}% (n={len(own):>3})  vs other {100*pt_:>3.0f}% (n={len(oth):>3})   "
                  f"gap {100*(po_-pt_):+.0f} pp")


def decode_full_column(recv):
    """Decode the 300 DeepSeek specs through one receiver `recv`. Used for the columns not present in
    receiver_replay.jsonl (mistrallarge) or replaced (devstral, now local). Resumable."""
    import random
    _sh = lambda sp: hashlib.sha256((sp or "").encode()).hexdigest()[:12]   # spec hash (same as section 1)
    items = RR.collect_encoded(); random.Random(0).shuffle(items); items = items[:300]
    done = set()
    if os.path.exists(GRID_OUT):
        for l in open(GRID_OUT):
            if l.strip():
                d = json.loads(l); done.add((d["recv"], d["sender"], d["seed"], d["secret"], d.get("spec_h", "")))
    todo = [it for it in items if (recv, "deepseekv4", it["seed"], it["secret"], _sh(it["spec"])) not in done]
    print(f"[{recv}<-deepseek] {len(todo)} to decode", flush=True)
    with open(GRID_OUT, "a") as fh, ThreadPoolExecutor(PAR) as ex:
        futs = {ex.submit(decode, recv, it["spec"], it["secret"]): it for it in todo}
        for i, f in enumerate(as_completed(futs), 1):
            it = futs[f]; authed, sub, _, _ = f.result()
            fh.write(json.dumps({"recv": recv, "sender": "deepseekv4", "seed": it["seed"],
                                 "secret": it["secret"], "spec_h": _sh(it["spec"]),
                                 "auth": authed, "submitted": sub[:60]}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  [{recv}<-deepseek] {i}/{len(todo)}", flush=True)


def main():
    gate_selftest()
    specs_by_sender = collect()
    specs = [s for v in specs_by_sender.values() for s in v]
    print(f"{len(specs)} rare-sender specs x {len(RECEIVERS)} receivers", flush=True)
    decode_new(specs)                      # rare specs x all receivers (devstral now local; rest resume-skip)
    decode_full_column("mistrallarge")     # deepseek x mistral-large  (exact same-model diagonal for Mistral)
    decode_full_column("devstral")         # deepseek x local devstral (one consistent host for the column)
    report()


if __name__ == "__main__":
    report() if os.getenv("REPORT_ONLY") else main()
