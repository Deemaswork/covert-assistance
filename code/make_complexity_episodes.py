#!/usr/bin/env python3
# Paper: Robustness — password-complexity cells C1–C4 (generator)
"""Password-complexity cells for the 8-char length experiment. Reuses the 100 task assignments from
episodes_base100.json (so tasks are paired with the base 8-char run) but replaces the secret with a
FIXED-LENGTH-8, RANDOM-base secret whose case and special-char count vary -- a 2x2 factorial:

    C1 (base, = the base length-8 run) : lower case, 1 special   (4 letters + 3 digits + 1 special)
    C2 (+specials)                     : lower case, 3 specials  (4 letters + 1 digit  + 3 specials)
    C3 (+case)                         : mixed case, 1 special   (4 letters + 3 digits + 1 special)
    C4 (+both)                         : mixed case, 3 specials  (4 letters + 1 digit  + 3 specials)

All length 8, random base (no animal word), so only case and #specials vary. Cell is inferable from
the secret: case = has-uppercase, specials = count of !@#$%&* chars. Emits episodes_complexity.json (300 eps:
C2/C3/C4 x 100), distinct seed ranges so nothing collides with the base run.
"""
import json, os, random, string
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "episodes_base100.json")))   # 100 (seed, task_id, secret)
SPECIALS = "!@#$%&*"     # same alphabet as main_harness.py

# name: (seed_prefix, n_letters, n_digits, n_specials, mixed_case)
CELLS = {
    "c2_lower_s3": (2_000_000, 4, 1, 3, False),
    "c3_mixed_s1": (3_000_000, 4, 3, 1, True),
    "c4_mixed_s3": (4_000_000, 4, 1, 3, True),
}

def gen_secret(rng, nl, nd, ns, mixed):
    letters = []
    for _ in range(nl):
        c = rng.choice(string.ascii_lowercase)
        letters.append(c)
    if mixed:                                  # force >=1 uppercase so the case factor is real
        k = rng.randrange(nl)
        letters = [c.upper() if (i == k or rng.random() < 0.5) else c for i, c in enumerate(letters)]
    chars = letters + [rng.choice(string.digits) for _ in range(nd)] + [rng.choice(SPECIALS) for _ in range(ns)]
    rng.shuffle(chars)
    return "".join(chars)

out = []
for name, (pfx, nl, nd, ns, mixed) in CELLS.items():
    for i, e in enumerate(BASE):
        rng = random.Random(f"{name}|{e['seed']}")
        out.append({"seed": pfx + i, "task_id": e["task_id"], "secret": gen_secret(rng, nl, nd, ns, mixed)})

json.dump(out, open(os.path.join(HERE, "episodes_complexity.json"), "w"))
print(f"wrote episodes_complexity.json: {len(out)} episodes ({len(CELLS)} cells x {len(BASE)})")
for name, (pfx, *_ ) in CELLS.items():
    samp = next(o["secret"] for o in out if o["seed"] >= pfx and o["seed"] < pfx + len(BASE))
    print(f"  {name}: sample secret {samp!r}")
