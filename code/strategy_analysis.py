#!/usr/bin/env python3
# Paper: Sequential structure — behavior over rounds (Appendix)
"""How the sender's approach evolves across rounds -- the behaviour behind the rate curve.

Section 4 shows the encoding rate rising with rounds and Section 5 catalogues which encodings exist,
but neither says what the sender is DOING over time. The interesting questions are sequential and the
per-round logs already answer them:

  PERSISTENCE   once the sender encodes, does it keep encoding?
  ESCALATION    a bounced round tells the sender only "your requirements did not reach the
                developer" -- it is never told a monitor exists or why. Does it back off, retry
                unchanged, or switch method?
  SWITCHING     when it does change method, is the change systematic (e.g. toward less legible
                encodings) or arbitrary?
  ONSET         is the round-2 jump a new strategy appearing, or the same strategy used more often?

The bounce case is the one that matters for oversight: a sender that responds to an unexplained
rejection by changing its encoding is doing black-box search against the monitor, without having been
told the monitor is there.

    python strategy_analysis.py
"""
import json, glob, os, collections, math

HERE = os.path.dirname(os.path.abspath(__file__))
from encoding_classifier import classify

# Families group the classifier's labels by what a monitor would have to recognise. The riddle
# labels are merged because their precision is poor individually (Section 5) but their union is the
# meaningful class: a payload recoverable only by resolving a description.
FAMILY = {"numeric": "codes", "concat-int": "codes",
          "base64": "cipher", "hex": "cipher", "rot13": "cipher", "reversed": "cipher",
          "riddle": "semantic", "riddle-indirect": "semantic"}
def fam(k):
    if k.startswith("caesar"): return "cipher"
    return FAMILY.get(k, None)


def load():
    """Per-episode sequence of (round, bounced, family, verbatim)."""
    eps = []
    for f in glob.glob(os.path.join(HERE, "runs_main_*.jsonl")):
        arm = os.path.basename(f).replace("runs_main_", "").replace(".jsonl", "")
        for line in open(f):
            if not line.strip().startswith("{"):
                continue
            r = json.loads(line)
            if "log" not in r:
                continue
            seq = []
            for e in r["log"]:
                k = classify(e.get("spec", "") or "", r["secret"])[0]
                seq.append({"round": e["round"], "bounced": int(e.get("flag", 0)),
                            "kind": k, "fam": fam(k), "auth": int(e.get("auth", 0))})
            eps.append({"arm": arm, "inv": arm.endswith("_inv"), "seed": r["seed"],
                        "covert": int(r.get("covert", 0) or 0), "seq": seq})
    return eps


def main():
    eps = load()
    print(f"{len(eps)} episodes, {sum(len(e['seq']) for e in eps)} rounds\n")

    # ---- ONSET: which round does an episode FIRST encode in? ----
    print("=== ONSET: first round in which the episode encodes ===")
    for g, lab in ((False, "no invitation"), (True, "invitation")):
        first = collections.Counter()
        n_enc = 0
        for e in eps:
            if e["inv"] != g:
                continue
            fr = next((s["round"] for s in e["seq"] if s["fam"]), None)
            if fr:
                first[fr] += 1; n_enc += 1
        tot = sum(1 for e in eps if e["inv"] == g)
        print(f"  {lab:<14} {n_enc}/{tot} episodes ever encode ({100*n_enc/tot:.1f}%); "
              f"first at round: " + " ".join(f"r{r}:{first[r]}" for r in range(1, 11) if first[r]))

    # ---- PERSISTENCE: P(encode next | encoded now), split by whether the round was bounced ----
    print("\n=== PERSISTENCE: P(encode in round r+1 | state at round r) ===")
    tab = collections.Counter()
    for e in eps:
        s = e["seq"]
        for i in range(len(s) - 1):
            enc_now = bool(s[i]["fam"]); nxt = bool(s[i + 1]["fam"])
            state = ("encoded" if enc_now else "clean") + ("/bounced" if s[i]["bounced"] else "/passed")
            tab[(e["inv"], state, nxt)] += 1
    for g, lab in ((False, "no invitation"), (True, "invitation")):
        print(f"  {lab}:")
        for state in ("clean/passed", "clean/bounced", "encoded/passed", "encoded/bounced"):
            a = tab[(g, state, True)]; b = tab[(g, state, False)]
            if a + b:
                print(f"    {state:<16} -> encodes next: {a:>5}/{a+b:<6} = {100*a/(a+b):5.1f}%")

    # ---- ESCALATION: what happens to the METHOD after a bounce ----
    print("\n=== AFTER AN ENCODED ROUND IS BOUNCED, what does the sender do next? ===")
    out = collections.Counter(); switch = collections.Counter()
    for e in eps:
        s = e["seq"]
        for i in range(len(s) - 1):
            if not (s[i]["fam"] and s[i]["bounced"]):
                continue
            nx = s[i + 1]
            if not nx["fam"]:
                out["stops encoding"] += 1
            elif nx["fam"] == s[i]["fam"]:
                out["same family"] += 1
            else:
                out["switches family"] += 1
                switch[f'{s[i]["fam"]} -> {nx["fam"]}'] += 1
    tot = sum(out.values())
    for k, v in out.most_common():
        print(f"    {k:<18} {v:>5}  {100*v/tot:5.1f}%")
    if switch:
        print("    switches:", ", ".join(f"{k} ({v})" for k, v in switch.most_common(6)))

    # control: what happens after an encoded round that PASSED the monitor
    print("\n=== ...and after an encoded round that PASSED (control) ===")
    out2 = collections.Counter()
    for e in eps:
        s = e["seq"]
        for i in range(len(s) - 1):
            if not (s[i]["fam"] and not s[i]["bounced"]):
                continue
            nx = s[i + 1]
            out2["stops encoding" if not nx["fam"] else
                 ("same family" if nx["fam"] == s[i]["fam"] else "switches family")] += 1
    t2 = sum(out2.values())
    for k, v in out2.most_common():
        print(f"    {k:<18} {v:>5}  {100*v/t2:5.1f}%")

    # two-proportion test on "switches family" after bounce vs after pass
    a1, n1 = out["switches family"], tot
    a2, n2 = out2["switches family"], t2
    if n1 and n2:
        p1, p2 = a1 / n1, a2 / n2; p = (a1 + a2) / (n1 + n2)
        z = (p1 - p2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        print(f"\n  switches family: {100*p1:.1f}% after bounce vs {100*p2:.1f}% after pass  "
              f"z={z:+.2f}  p={math.erfc(abs(z)/math.sqrt(2)):.4f}")

    # ---- how many distinct families does an episode try? ----
    print("\n=== METHOD DIVERSITY within an episode ===")
    for g, lab in ((False, "no invitation"), (True, "invitation")):
        d = collections.Counter()
        for e in eps:
            if e["inv"] != g: continue
            fams = {s["fam"] for s in e["seq"] if s["fam"]}
            if fams: d[len(fams)] += 1
        t = sum(d.values())
        if t:
            print(f"  {lab:<14} " + "  ".join(f"{k} famil{'y' if k==1 else 'ies'}: {v} ({100*v/t:.0f}%)"
                                              for k, v in sorted(d.items())))

    # ---- trajectory of the covert successes ----
    print("\n=== THE COVERT SUCCESSES ===")
    wins = [e for e in eps if e["covert"]]
    if not wins:
        print("  (no covert successes in this data)")
    else:
        rounds = [max(s["round"] for s in e["seq"]) for e in wins]
        prior_bounce = sum(1 for e in wins if any(s["bounced"] for s in e["seq"][:-1]))
        prior_enc = sum(1 for e in wins if any(s["fam"] for s in e["seq"][:-1]))
        print(f"  n={len(wins)}; winning round median={sorted(rounds)[len(rounds)//2]}, "
              f"range {min(rounds)}-{max(rounds)}")
        print(f"  had a bounced round before the win: {prior_bounce}/{len(wins)} "
              f"({100*prior_bounce/len(wins):.0f}%)")
        print(f"  had an earlier encoding attempt:    {prior_enc}/{len(wins)} "
              f"({100*prior_enc/len(wins):.0f}%)")


if __name__ == "__main__":
    main()
