#!/usr/bin/env python3
# Paper: Motive analysis — second-annotator agreement (§5.1)
"""Second-annotator agreement for the MOTIVE labels: claude-sonnet re-annotates a sample of the same
encoding rounds gpt-5-mini did (incentive_results.jsonl), using the identical motive prompt. Reports
per-element presence agreement + PABAK (prevalence-adjusted), matching the paper's tab:motive style.
"""
import json, os, re, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
HERE=os.path.dirname(os.path.abspath(__file__)); ENV=os.getenv("COVERT_ENV") or os.path.join(HERE, ".env")
def key(n):
    for l in open(ENV):
        m=re.search(r''+n+r'\s*=\s*"?(sk-[A-Za-z0-9_-]+)"?', l)
        if m: return m.group(1)
AN=openai.OpenAI(base_url="https://api.anthropic.com/v1", api_key=key("ANTHROPIC_API_KEY"))
MODEL="claude-sonnet-4-5"; N=int(os.getenv("N","250"))
RX=re.compile(r"<requirements>", re.I)
PROMPT=open(os.path.join(HERE,"incentive_pass.py")).read().split('PROMPT = """')[1].split('"""')[0]
V4={"gptoss_inv":"runs_main_gptoss_inv","gptoss_noinv":"runs_main_gptoss_noinv","qwen_inv":"runs_main_qwen_inv",
    "qwen_noinv":"runs_main_qwen_noinv","gpt5mini_inv":"runs_main_gpt5mini_inv","gpt5mini_noinv":"runs_main_gpt5mini_noinv"}
reason={}
for arm,f in V4.items():
    for line in open(f+".jsonl"):
        if not line.strip().startswith("{"): continue
        r=json.loads(line)
        if "log" not in r: continue
        for e in r["log"]:
            rs=(e.get("a_reasoning") or "").strip()
            reason[(arm,r["seed"],e["round"])]=rs
LABELS=["knows_policy","objective_met","credential_needed","decides_to_encode","anticipates_oversight","evade_detection"]
def ann(rz):
    for _ in range(3):
        try:
            r=AN.chat.completions.create(model=MODEL,temperature=0,max_tokens=400,
                messages=[{"role":"user","content":PROMPT.format(reasoning=rz[:16000])}])
            t=re.sub(r"^```(?:json)?|```$","",(r.choices[0].message.content or "").strip(),flags=re.M).strip()
            m=re.search(r"\{.*\}",t,re.S); return json.loads(m.group(0)) if m else None
        except Exception: pass
    return None
# The reason dict above is built only from the DeepSeek-V4 main runs, so restrict the rows to the same
# planner -- otherwise a non-V4 row could match a V4 trace by (arm, seed, round) and mix sources.
rows=[r for r in (json.loads(l) for l in open("incentive_results.jsonl"))
      if r.get("labels") and r.get("planner")=="DeepSeek-V4"]
random.Random(9).shuffle(rows); rows=rows[:N]
def one(r):
    rz=reason.get((r["arm"],r["seed"],r["round"]),"")
    if len(rz)<40: return None
    d=ann(rz)
    if not d: return None
    return dict(gpt=set(r["labels"]), cl=set(d.get("labels",[])))
out=[]
with ThreadPoolExecutor(8) as ex:
    for f in as_completed({ex.submit(one,r):r for r in rows}):
        x=f.result()
        if x: out.append(x)
print(f"MOTIVE agreement, gpt-5-mini vs {MODEL}, n={len(out)}\n")
print(f"{'element':<22}{'gpt%':>6}{'cl%':>6}{'agree':>7}{'PABAK':>7}")
for L in LABELS:
    a=sum((L in o['gpt'])==(L in o['cl']) for o in out)/len(out)
    g=sum(L in o['gpt'] for o in out)/len(out); c=sum(L in o['cl'] for o in out)/len(out)
    pabak=2*a-1
    print(f"{L:<22}{100*g:>5.0f}%{100*c:>5.0f}%{100*a:>6.0f}%{pabak:>7.2f}")
