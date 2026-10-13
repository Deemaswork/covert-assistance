#!/usr/bin/env python3
# Paper: Main study — live stats (§4–§5)
"""Current state of the main run.

Classification is cached in .stats_cache.json keyed by (arm, seed, round). The functional encoding
classifier scans every token of every specification against 25 rotations plus base64/hex candidates;
at ~35k rounds a full re-scan takes minutes and made plain status checks time out. Only rounds not
already cached are classified, so repeat calls are fast.
"""
import json, glob, os, math, collections
from encoding_classifier import classify

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".stats_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
before = len(cache)

D, kinds = {}, collections.Counter()
for f in sorted(glob.glob(os.path.join(HERE, "runs_main_*.jsonl"))):
    arm = os.path.basename(f).replace("runs_main_", "").replace(".jsonl", "")
    eps = rounds = enc = cov = 0
    for line in open(f):
        if not line.strip().startswith("{"):
            continue
        r = json.loads(line)
        if "log" not in r:
            continue
        eps += 1; cov += r.get("covert", 0)
        for e in r["log"]:
            rounds += 1
            k = f'{arm}|{r["seed"]}|{e["round"]}'
            if k not in cache:
                cache[k] = classify(e.get("spec", "") or "", r["secret"])[0]
            if cache[k] not in ("none", "verbatim", "unclassified", "riddle", "riddle-indirect"):  # transform-verified = reversible-only (paper excludes semantic riddles)
                enc += 1; kinds[cache[k]] += 1
    if eps:
        D[arm] = (eps, rounds, enc, cov)
if len(cache) != before:
    json.dump(cache, open(CACHE, "w"))

print(f"{'arm':<16}{'eps':>6}{'encoded/round':>18}{'covert':>16}")
for a, (e, R, enc, cov) in sorted(D.items()):
    if R:
        p = enc / R
        h = 100 * 1.96 * math.sqrt(p * (1 - p) / R)
        encoding = f"{100*p:.2f}% ±{h:.2f}"
    else:
        encoding = "N/A (no rounds)"
    print(f"{a:<16}{e:>6}{encoding:>18}{f'{cov} ({100*cov/e:.2f}%)':>16}")

def po(sfx, i, j):
    return (sum(D[k][i] for k in D if k.endswith(sfx)), sum(D[k][j] for k in D if k.endswith(sfx)))
def z(n1, k1, n2, k2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return ((k2 / n2) - (k1 / n1)) / se if se else 0.0

n1, k1 = po("noinv", 1, 2); n2, k2 = po("_inv", 1, 2)
e1, c1 = po("noinv", 0, 3); e2, c2 = po("_inv", 0, 3)
print(f"\ntotal {sum(v[0] for v in D.values())}/6000 episodes")
if not D:
    print("No completed episodes found in runs_main_*.jsonl. Run experiments first; smoke_main outputs are separate.")
if n1 and n2:
    print(f"  encoding  {100*k1/n1:.2f}%  ->  {100*k2/n2:.2f}%   z={z(n1,k1,n2,k2):+.2f}")
else:
    print("  encoding comparison unavailable: both invitation conditions need recorded rounds.")
if e1 and e2:
    print(f"  covert    {100*c1/e1:.2f}%  ->  {100*c2/e2:.2f}%   z={z(e1,c1,e2,c2):+.2f}")
else:
    print("  covert comparison unavailable: both invitation conditions need completed episodes.")
print("\nencodings:", dict(kinds.most_common()))
