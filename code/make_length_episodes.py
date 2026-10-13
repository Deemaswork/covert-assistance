#!/usr/bin/env python3
# Paper: Robustness — password-length cells L12/L16 (generator)
"""C4-at-longer-length cells. C4 = mixed case, 1 digit, 3 specials (the hardest complexity cell). Here
we hold that recipe FIXED and vary ONLY length by adding letters, so length is the sole manipulated
variable (vs the C1-C4 factorial which varied case/#specials at fixed length 8):

    L12 : mixed case, 8 letters + 1 digit + 3 specials  (len 12)
    L16 : mixed case, 12 letters + 1 digit + 3 specials (len 16)

Reuses the 100 task assignments from episodes_base100.json (same tasks as C1-C4). Distinct seed
prefixes (12e6, 16e6) so nothing collides. Emits episodes_length.json (200 eps)."""
import json, os, random, string
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "episodes_base100.json")))   # 100 (seed, task_id, secret)
SPECIALS = "!@#$%&*"

# name: (seed_prefix, length)  -- composition is C4 held fixed: 1 digit, 3 specials, rest letters, mixed
CELLS = {"l12_c4": (12_000_000, 12), "l16_c4": (16_000_000, 16)}
N_DIGITS, N_SPECIALS = 1, 3

def gen_secret(rng, length):
    nl = length - N_DIGITS - N_SPECIALS
    letters = [rng.choice(string.ascii_lowercase) for _ in range(nl)]
    k = rng.randrange(nl)                                   # force >=1 uppercase (case factor real)
    letters = [c.upper() if (i == k or rng.random() < 0.5) else c for i, c in enumerate(letters)]
    chars = letters + [rng.choice(string.digits) for _ in range(N_DIGITS)] \
                    + [rng.choice(SPECIALS) for _ in range(N_SPECIALS)]
    rng.shuffle(chars)
    return "".join(chars)

out = []
for name, (pfx, length) in CELLS.items():
    for i, e in enumerate(BASE):
        rng = random.Random(f"{name}|{e['seed']}")
        out.append({"seed": pfx + i, "task_id": e["task_id"], "secret": gen_secret(rng, length)})

json.dump(out, open(os.path.join(HERE, "episodes_length.json"), "w"))
print(f"wrote episodes_length.json: {len(out)} episodes ({len(CELLS)} cells x {len(BASE)})")
for name, (pfx, length) in CELLS.items():
    samp = next(o["secret"] for o in out if pfx <= o["seed"] < pfx + len(BASE))
    print(f"  {name} (len {length}): sample secret {samp!r}  (len {len(samp)})")
