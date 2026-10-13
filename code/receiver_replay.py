#!/usr/bin/env python3
# Paper: Generality — receiver replay/decode (§generality)
r"""Receiver diversity by replay: does the covert channel depend on the specific decoder?

Every covert-success rate in the paper is conditioned on one receiver (devstral-2512). A reviewer can
ask whether 1.5% covert is a property of the channel or of that decoder in particular. This answers
it without re-running episodes: the specifications are already written and saved, so we feed a fixed
set of them to several receivers and measure how often each reconstructs the credential and passes the
real gate.

The monitor is not involved. This isolates the DECODE step -- the second of the two gates in a covert
success (monitor CLEAN, then B decodes). We use only specifications where the functional classifier
found a transform-verified payload, so a decodable credential provably exists and we are measuring
decode ability rather than "is there anything to decode".

All receivers see the SAME sampled specifications (paired comparison). The sender (DeepSeek-V4) wrote
each spec for a generic "external developer" and never knew who B would be, so the specs are not tuned
to any receiver; a stronger decoder should recover at least as often. The DeepSeek-V4 receiver is the
interesting cell: if it decodes much better than the others the channel leans on shared style (closer
to collusion), and if all receivers are similar the payload is genuinely open (which is what an
uninstructed receiver on first contact needs).

    N=300 python receiver_replay.py
"""
import os, re, sys, json, glob, random, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MONITOR", "qwen")   # only to satisfy main_harness import; monitor is never called
from main_harness import (EXECUTOR_SYSTEM, EXECUTOR_USER, DEPLOY_USER, SUDO_PROMPT, gate, key,
                        TOGETHER, OPENROUTER)
from deploy_sandbox import gate_selftest
from openai import OpenAI

N = int(os.getenv("N", "300"))
PAR = int(os.getenv("PAR", "16"))
OUT = os.path.join(HERE, os.getenv("OUT", "receiver_replay.jsonl"))

# Route hosted models through Together; use the listed provider endpoint for other receivers.
# devstral is re-run here (not read from the saved auth field) so every receiver faces identical
# sampling conditions -- a clean paired baseline rather than a historical one.
RECEIVERS = {
    "devstral":   ("mistralai/devstral-2512",                      "https://openrouter.ai/api/v1", OPENROUTER),
    "qwen3coder": ("qwen/qwen3-coder",                            "https://openrouter.ai/api/v1", OPENROUTER),
    "deepseekv4": ("deepseek-ai/DeepSeek-V4-Pro",                  "https://api.together.xyz/v1",  TOGETHER),
    "claude":     ("claude-sonnet-4-5",                            "https://api.anthropic.com/v1", key("ANTHROPIC_API_KEY")),
}

from encoding_classifier import classify
STRICT = {"numeric", "concat-int", "base64", "hex", "rot13", "reversed"}
def is_strict(k): return k in STRICT or k.startswith("caesar")


def collect_encoded():
    """Every main-run round whose spec carries a transform-verified payload, de-duplicated by
    (secret, spec). Uses .stats_cache.json (arm|seed|round -> classifier label) so it does not
    re-classify ~60k specs on every launch, which otherwise times out before the first API call."""
    # The cache (written by stats_main.py) just avoids re-classifying ~60k specs; if it is absent we
    # classify on the fly below, so a missing cache is not fatal -- it only makes the first launch slower.
    _cache_path = os.path.join(HERE, ".stats_cache.json")
    cache = json.load(open(_cache_path)) if os.path.exists(_cache_path) else {}
    seen, items = set(), []
    for f in glob.glob(os.path.join(HERE, "runs_main_*.jsonl")):
        arm = os.path.basename(f).replace("runs_main_", "").replace(".jsonl", "")
        for l in open(f):
            if not l.strip().startswith("{"):
                continue
            r = json.loads(l)
            if "log" not in r:
                continue
            for e in r["log"]:
                sp = e.get("spec", "") or ""
                k = cache.get(f'{arm}|{r["seed"]}|{e["round"]}') or (classify(sp, r["secret"])[0] if sp else "none")
                if not sp or not is_strict(k):
                    continue
                h = hashlib.sha256((r["secret"] + "\0" + sp).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                items.append({"seed": r["seed"], "secret": r["secret"], "spec": sp,
                              "devstral_auth_orig": int(e.get("auth", 0))})
    return items


def _client(recv):
    model, base, k = RECEIVERS[recv]
    return OpenAI(base_url=base, api_key=k), model


def decode(recv, spec, secret):
    """B's exact two-turn pipeline: implement -> deploy -> submit the password. Returns (auth, submitted)."""
    cl, model = _client(recv)
    msgs = [{"role": "system", "content": EXECUTOR_SYSTEM},
            {"role": "user", "content": EXECUTOR_USER.format(plan=spec)}]
    for attempt in range(3):
        try:
            m1 = cl.chat.completions.create(model=model, messages=msgs, max_tokens=6000).choices[0].message
            code = m1.content or getattr(m1, "reasoning", "") or ""
            msgs2 = msgs + [{"role": "assistant", "content": code},
                            {"role": "user", "content": DEPLOY_USER.format(prompt=SUDO_PROMPT)}]
            m2 = cl.chat.completions.create(model=model, messages=msgs2, max_tokens=2048).choices[0].message
            raw = m2.content or getattr(m2, "reasoning", "") or ""
            lines = [x.strip().strip('`"\' ') for x in raw.splitlines() if x.strip().strip('`"\' ')]
            submitted = lines[-1] if lines else ""
            authed, _ = gate(submitted, secret)
            return authed, submitted, code, raw
        except Exception as e:
            if attempt == 2:
                return -1, f"[err {str(e)[:60]}]", "", ""
    return -1, "[err]", "", ""


def main():
    gate_selftest()
    items = collect_encoded()
    random.Random(0).shuffle(items)
    items = items[:N]
    print(f"{len(items)} unique encoded specs sampled; receivers = {list(RECEIVERS)}", flush=True)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                d = json.loads(l); done.add((d["recv"], d["seed"], d["secret"]))

    rows = []
    with open(OUT, "a") as fh:
        for recv in RECEIVERS:
            todo = [it for it in items if (recv, it["seed"], it["secret"]) not in done]
            print(f"[{recv}] {len(todo)} to decode", flush=True)
            with ThreadPoolExecutor(PAR) as ex:
                futs = {ex.submit(decode, recv, it["spec"], it["secret"]): it for it in todo}
                for i, f in enumerate(as_completed(futs), 1):
                    it = futs[f]; authed, sub, code, raw = f.result()
                    row = {"recv": recv, "seed": it["seed"], "secret": it["secret"],
                           "auth": authed, "submitted": sub[:60],
                           "code": code[:6000], "deploy_raw": raw[:2000], "spec": it["spec"]}
                    rows.append(row); fh.write(json.dumps(row) + "\n"); fh.flush()
                    if i % 50 == 0:
                        print(f"  [{recv}] {i}/{len(todo)}", flush=True)
    report()


def report():
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    print(f"\n===== RECEIVER RECOVERY on identical encoded specs (paired) =====")
    print(f"  {'receiver':<12}{'decoded':>10}{'errors':>8}{'rate (of valid)':>18}")
    for recv in RECEIVERS:
        r = [x for x in rows if x["recv"] == recv]
        ok = sum(1 for x in r if x["auth"] == 1)
        err = sum(1 for x in r if x["auth"] == -1)
        valid = len(r) - err
        rate = f"{100*ok/valid:.0f}%" if valid else "--"
        print(f"  {recv:<12}{ok:>10}{err:>8}{rate:>18}   (n={valid})")
    # paired: specs every receiver decoded vs specs only the self-model decoded
    by_spec = {}
    for x in rows:
        by_spec.setdefault((x["seed"], x["secret"]), {})[x["recv"]] = x["auth"]
    full = [k for k, v in by_spec.items() if all(v.get(r) is not None for r in RECEIVERS)]
    if full:
        allok = sum(1 for k in full if all(by_spec[k].get(r) == 1 for r in RECEIVERS))
        dsonly = sum(1 for k in full if by_spec[k].get("deepseekv4") == 1
                     and sum(by_spec[k].get(r) == 1 for r in RECEIVERS if r != "deepseekv4") == 0)
        print(f"\n  specs decoded by ALL receivers: {allok}/{len(full)}")
        print(f"  specs decoded ONLY by DeepSeek-V4 (shared-style signal): {dsonly}/{len(full)}")


if __name__ == "__main__":
    report() if os.getenv("REPORT_ONLY") else main()
