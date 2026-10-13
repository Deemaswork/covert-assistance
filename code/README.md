# Covert Assistance

Code for **Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems**
(Alnuhait, Wang, Khalifa, Peng, 2026).

[![arXiv](../assets/arxiv-badge.svg)](https://arxiv.org/abs/2609.39050)
[![Project page](../assets/project-badge.svg)](https://deemaswork.github.io/covert-assistance/)
[![Setup and code guide](../assets/setup-badge.svg)](#setup)

## Setup

Use Linux with Python 3.11–3.13, Docker, and permission to run `sudo -n docker`.
The Python dependency versions below match the existing experiment environment;
model availability and provider access must be checked with your accounts.
The harness executes model-generated Python locally as well as deployment scripts
inside containers. Run experiments in a disposable research VM without unrelated credentials.

From the repository root:

```bash
cd code
python3 -m venv ag_venv
source ag_venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally to supply your provider keys. The main harness and several analysis
scripts read this file directly; exporting API keys alone is insufficient.
Set `COVERT_ENV=/absolute/path/to/.env` to use another file. Never commit real keys.
The small example below needs Together and OpenRouter access. OpenAI is needed for the
`gpt5mini` monitor and OpenAI-based judging/paraphrasing; Anthropic and Baseten are optional.
Model identifiers and endpoint overrides are defined in `main_harness.py`.

From the same `code/` directory, build the deployment image:

```bash
sudo -n docker build -t deploy-host:latest docker_deploy
```

If using snap Docker, copy the build directory beneath your home directory and run the
build there. Runtime mounts default to `~/dmg_work` for the same reason.
Verify the password gate without any model/API calls (from `code/`, with the venv active):

```bash
python -c "from deploy_sandbox import gate_selftest; gate_selftest(); print('Gate checks passed')"
```

## Quickstart: one episode

From `code/`, with the virtual environment active and `.env` configured:

```bash
MONITORS=gptoss INVITES=0 LIMIT=1 PAR=1 DEPLOY_MAX_CONTAINERS=1 \
  PREFIX=smoke_main python run_main.py
```

This runs one frozen episode with one monitor and no invitation clause. It makes paid
API calls and may take multiple rounds; no fixed cost is guaranteed. Results go to
`smoke_main_gptoss_noinv.jsonl`, separate from main-study outputs. A record containing
`log` is a completed episode; check `error`, `noresult`, or `timeout` records before
interpreting results. Re-running resumes completed seeds.

## Full experiments and analysis

`python run_main.py` defaults to **6,000 episodes** (three monitors × two invitation
conditions × 1,000 episodes), with `PAR=60`. The default Qwen monitor additionally
requires a local OpenAI-compatible endpoint; configure `QWEN_URLS` or select only
available monitors with `MONITORS`. These are research-scale runs with paid API usage.
The runner uses the active Python interpreter; `PY` can override it.

After collecting both invitation conditions, `python stats_main.py` summarizes
main-study outputs. `python encoding_judge_full.py` performs additional paid judge calls
on the run files listed in that script. Sample records are format examples, not enough
to reproduce the paper's aggregate results.

Conditions are environment flags in `main_harness.py` (monitor, defenses, intent rule,
recipient framing, credential provenance, and other ablations). The build-ladder
experiments additionally require host C/C++ build tools and Python packaging tools
(`build`, `setuptools`, `wheel`); see `prod_leak.py` and `costdial_env.py`.

## Code map

| Part of the paper | Script(s) |
|---|---|
| Main study harness (§4–§5) | `main_harness.py`, runner `run_main.py`, live stats `stats_main.py` |
| Encoding detection | `encoding_classifier.py` (transform floor), `encoding_judge_full.py` (judge-corrected) |
| Defenses | `defense_on_wins.py` (provisioning replay on the 54 wins); older table via `aware_monitor.py` + `legacy_harness.py` |
| Build ladder, L1–L4 (§5.2.3) | `prod_leak.py` |
| Receiver / generality | `two_way_experiment.py`, `receiver_grid.py`, `receiver_replay.py` |
| Motive analysis (§5.1) | `label_traces.py`, `incentive_pass.py`, `motive_agreement.py`, `relabel_needs.py`, `strategy_analysis.py` |
| Prompt appendix (generated) | `dump_prompts_tex.py` |

Support modules: `deploy_sandbox.py` (the real sudo gate in a container), `judge_specs.py`
and `costdial_env.py` (build-ladder helpers reused by `prod_leak.py`), `judge_encoding.py`
(LLM-judge validation of the encoding classifier).

Inputs:

| File | What it is |
|---|---|
| `tasks_deploy/tasks_train.json` | The task dataset: 100 verified Python coding tasks (`id`, `statement`, `reference`, `cases`), the pool the harness draws each episode's task from. |
| `grid_specs.json` | 76 saved sender-generated requirements (49 Mistral, 27 Claude) for the receiver-generality grid. |
| `episodes_frozen_v1.json` | Frozen episode set used for replay. |

## Data

The full task dataset ships in `tasks_deploy/` (see the table above). Full *run* outputs are
large and are regenerated by the harness; `sample_data/` holds a few truncated records showing
the output format.
