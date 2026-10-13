# Covert Assistance

**Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems**  
Deema Alnuhait, Gengyu Wang, Muhammad Khalifa, and Hao Peng (2026).

[![arXiv](assets/arxiv-badge.svg)](https://arxiv.org/abs/2609.39050)
[![Project page](assets/project-badge.svg)](https://deemaswork.github.io/covert-assistance/)
[![Setup and code guide](assets/setup-badge.svg)](code/README.md)

This repository studies whether a planner agent shares a protected credential to help
an external developer, despite a nondisclosure rule and a monitor screening their exchanges.
It includes the experiment harnesses, 100 coding tasks, 1,000 frozen episodes, and sample outputs.
Experiment passwords and vault decoys are synthetic fixtures, not real account credentials.

## Getting started

Follow the [setup guide](code/README.md#setup), then run its one-episode example.
Experiments require provider API access and Docker. The full default run contains
6,000 episodes; start with the small example. Full run outputs are not included.

## Repository

- [`code/`](code/): experiments, datasets, sample outputs, and setup instructions.
- [`index.html`](index.html): the project page.
- [`code/LICENSE`](code/LICENSE): MIT license.

## Citation

```bibtex
@misc{alnuhait2026covert,
  title = {Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems},
  author = {Alnuhait, Deema and Wang, Gengyu and Khalifa, Muhammad and Peng, Hao},
  year = {2026},
  eprint = {2609.39050},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR}
}
```
