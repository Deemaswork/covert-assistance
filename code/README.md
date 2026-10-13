# Covert Assistance

Code for **Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems**
(Alnuhait, Wang, Khalifa, Peng, 2026).

[![arXiv](https://raw.githubusercontent.com/Deemaswork/covert-assistance/main/assets/arxiv-badge.svg)](https://arxiv.org/abs/2609.39050)
[![Project page](https://raw.githubusercontent.com/Deemaswork/covert-assistance/main/assets/project-badge.svg)](https://deemaswork.github.io/covert-assistance/)
[![Setup and code guide](https://raw.githubusercontent.com/Deemaswork/covert-assistance/main/assets/setup-badge.svg)](#setup)

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

Most conditions are environment flags read by `main_harness.py`; set the flag in the
environment and run the same `run_main.py` (it passes the environment through to the
harness), then run the analysis script. The table below maps each paper result to its
command. All commands assume you are in `code/` with the venv active and `.env` configured,
and append the main-study runner flags (`MONITORS=... INVITES=... PREFIX=... PAR=...`) as in
the Quickstart.

### Reproducing each experiment

| Experiment (paper) | How to run it |
|---|---|
| Main study, 6,000 episodes | `MONITORS=gptoss,qwen,gpt5mini INVITES=0,1 PREFIX=runs_main python run_main.py`, then `python stats_main.py` and `python encoding_judge_full.py` |
| Oversight ablation (§5) | `NOMON=1` drops the LLM monitor (keeps the DLP filter); `NOMON=1 NODLP=1` disables oversight entirely (also drops the verbatim DLP filter) |
| Human-vs-agent framing | add `CHAT_DEV=1` (chat framing); `HUMAN_DEV=1` presents the recipient as a human |
| Stronger (intent) rule | add `EFFECT_POLICY=1` |
| Recipient distrust | add `ADV_DEV=1` |
| Credential provenance | add `VAULT=1` (retrieved from a vault); `SENSLABEL=1` / `NEUTRALTOOL=1` are the framing variants |
| Healthcare / SSN domain | add `PII=1` |
| In-loop defenses | add `AWARE_MON=1` (monitor-aware priming) or `PARA=1` (paraphrase) |
| Secret complexity (C1–C4) | C1 is the base length-8 run (`EPISODES=episodes_base100.json python run_main.py`); C2–C4: `python make_complexity_episodes.py`, then `EPISODES=episodes_complexity.json python run_main.py` |
| Secret length (L12/L16) | `python make_length_episodes.py`, then `EPISODES=episodes_length.json python run_main.py` |
| Build ladder L1–L4 (§5.2.3) | `LEVEL=1 python prod_leak.py` … `LEVEL=4 python prod_leak.py` |
| Provisioning-replay defense | `python defense_on_wins.py` (54 wins); earlier-harness table: `python aware_monitor.py` |
| Receiver generality | `python receiver_replay.py`, then `python receiver_grid.py` |
| Two-way (receiver replies) | `MSG_SRC=deploy` (forward B's answer), `decide` (status update), or `optional` (B may stay silent): `MSG_SRC=optional python two_way_experiment.py <seed>` |
| Motive codebook (§5.1) | `python incentive_pass.py`, then `python theme_converge.py` → `python theme_final.py` → `python codebook_alone.py` |
| Leakage detectors (Appendix) | `python judge_encoding.py`, then `python detector_eval.py` (it selects the gpt-oss labels itself; needs torch/transformers + GPU) |
| Sequential structure (Appendix) | `python strategy_analysis.py` |

The build-ladder experiments additionally require host C/C++ build tools and Python
packaging tools (`build`, `setuptools`, `wheel`); see `prod_leak.py` and `costdial_env.py`.
Analysis scripts regenerate their own intermediate files from the run outputs, so only the
run outputs (and `.env`) are needed — no precomputed data ships with the release.

## Code map

| Part of the paper | Script(s) |
|---|---|
| Main study harness (§4–§5) | `main_harness.py`, runner `run_main.py`, live stats `stats_main.py` |
| Encoding detection | `encoding_classifier.py` (transform floor), `encoding_judge_full.py` (judge-corrected) |
| Defenses | `defense_on_wins.py` (provisioning replay on the 54 wins); older table via `aware_monitor.py` + `legacy_harness.py` |
| Build ladder, L1–L4 (§5.2.3) | `prod_leak.py` |
| Receiver / generality | `two_way_experiment.py`, `receiver_grid.py`, `receiver_replay.py` |
| Motive analysis (§5.1) | `label_traces.py`, `incentive_pass.py`, `motive_agreement.py`, `relabel_needs.py`, `strategy_analysis.py` |
| Motive — bottom-up codebook (§5.1) | `theme_converge.py` (discovery) → `theme_final.py` (merge into 8 concepts + 2 directed questions) → `codebook_alone.py` (each-alone scoring). Run after `run_main.py` and `incentive_pass.py`; intermediates (`converge_out.json`, `final_out.json`) are regenerated, not shipped. |
| Secret complexity/length (robustness) | `make_complexity_episodes.py` (C1–C4), `make_length_episodes.py` (L12/L16); run via `EPISODES=<file> python run_main.py` |
| Leakage-detector baselines (Appendix) | `detector_eval.py` — an entropy-collapse detector implementing PRISM's signal, plus a NARCBench-style activation probe, over Qwen3-32B. This reimplements the signals and runs them on a proxy open model (the closed Planner's own activations are unavailable); it is not PRISM's own package. Reads `judge_encoding.jsonl` (run `judge_encoding.py` on the gpt-oss files). Needs extra deps: `torch`, `transformers`, `scikit-learn`, `numpy`, and a GPU. |
| Prompt appendix (generated) | `dump_prompts_tex.py` |

Support modules: `deploy_sandbox.py` (the real sudo gate in a container), `judge_specs.py`
and `costdial_env.py` (build-ladder helpers reused by `prod_leak.py`), `judge_encoding.py`
(LLM-judge validation of the encoding classifier).

Inputs:

| File | What it is |
|---|---|
| `tasks_deploy/tasks_train.json` | The task dataset: 100 verified Python coding tasks (`id`, `statement`, `reference`, `cases`), the pool the harness draws each episode's task from. |
| `grid_specs.json` | 76 saved sender-generated requirements (49 Mistral, 27 Claude) for the receiver-generality grid. |
| `episodes_frozen_v1.json` | The 1,000 frozen `(seed, task_id, secret)` episodes the main study runs (`EPISODES` default). |
| `episodes_base100.json` | 100 base `(seed, task_id, secret)` episodes (length-8 secrets) that the complexity/length generators build on. |
| `episodes_complexity.json` | 300 episodes (C2/C3/C4 × 100), length-8 secrets varying case and special-char count; produced by `make_complexity_episodes.py`. |
| `episodes_length.json` | 200 episodes (L12/L16 × 100), the hardest complexity recipe held fixed while only length varies; produced by `make_length_episodes.py`. |

## Data

The full task dataset ships in `tasks_deploy/` (see the table above). Full *run* outputs are
large and are regenerated by the harness; `sample_data/` holds a few truncated records showing
the output format.
