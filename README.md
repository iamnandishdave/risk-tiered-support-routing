# Risk-Tiered Authority Allocation for Human–AI Customer Support

[![DOI](https://zenodo.org/badge/1415157619.svg)](https://doi.org/10.5281/zenodo.23334096)

Code and data for the paper:

> **Human–AI Collaborative Customer Support for Digital Asset Platforms: A Trust-Aware Framework for Security, Explainability, and Service Reliability**
> Nandish Dave. *IEEE Access* (under review), 2026.

The paper argues that on digital asset platforms, customer support is a security
boundary, and that an AI assistant's authority should be set by the **consequence,
uncertainty and irreversibility** of a request rather than by the AI's confidence.
This repository reproduces the results and figures in Section V of the paper.

The version used for the paper is archived on Zenodo as release v1.0:
[doi.org/10.5281/zenodo.23334097](https://doi.org/10.5281/zenodo.23334097).

## Main result

With thresholds selected on a held-out validation split and applied once to the
untouched test set (n = 3,080):

| Policy | Automation | Unsafe automation, Tier 2–3 |
|---|---|---|
| P0 confidence-only | 53.3% | 52.3% |
| P1 predicted-intent tier | 32.6% | 0.92% |
| P2 risk-conservative tier | 32.9% | 0.71% |

Confidence-only routing leaves about half of high-consequence requests under AI
authority despite 91.5% classifier accuracy; tier-based routing reduces this to
under 1%.

## Repository contents

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── train.csv                          Banking77 training split (10,003 queries)
│   └── test.csv                           Banking77 test split (3,080 queries)
├── intent_coding.csv                      C, U, I scores and tier for all 77 intents
├── Supplementary_S1_Intent_Coding.xlsx    Same coding, with columns for independent coders
├── analysis.py                            Fig. 3 and the fixed-threshold results
├── analysis2.py                           Validation-selected results (Tables 4–7)
├── finetuned_transformer_baseline.py      Optional BERT/RoBERTa classifier (GPU recommended)
├── llm_zeroshot_baseline.py               Optional zero-shot LLM classifier (API key required)
└── results/                               Output files from the scripts above, including Fig. 3
```

## Setup

Python 3.11 is recommended.

```bash
git clone https://github.com/iamnandishdave/risk-tiered-support-routing.git
cd risk-tiered-support-routing
pip install -r requirements.txt
```

## Reproducing the results

Both scripts run in a few minutes on a laptop CPU. The random seed is fixed at 42.
Results were produced with the exact package versions in `requirements.txt`; other
scikit-learn versions may give slightly different numbers.

### Tables 4–7 (validation-selected thresholds)

```bash
python analysis2.py
```

The first lines of output should read:

```
train_fit=8002  val=2001  test=3080 (test untouched until step 4)
Selected on validation set (not test): P0 tau=0.95, P1 tau=0.57, P2 tau=0.5 theta=0.2
Final test accuracy (refit on full train): 0.9153
```

| Output | Paper |
|---|---|
| `results_valtuned_main.csv` | Table 4 (under-tiering rates are printed to the console) |
| `results_paired_bootstrap.csv` | Table 5 |
| `results_additional_baselines.csv` | Table 6 |
| `results_expected_cost.csv` | Table 7 |
| `results_cost_sensitivity.csv` | Cost sensitivity, Section V-D |
| `results_calibration.csv` | Expected calibration error, Section V-C |
| `results_random_retiering.csv` | Random re-tiering control, Section V-C |
| `results_aggregation_robustness.csv` | Alternative aggregation rules, Section V-C |
| `selected_params.json` | Thresholds selected on the validation set |

### Fig. 3 and fixed-threshold results

```bash
python analysis.py
```

| Output | Content |
|---|---|
| `figure3_threshold_sweep.png` | Fig. 3: effect of the confidence threshold τ |
| `results_main.csv` | All three policies at fixed τ = 0.8, θ = 0.10, with 95% bootstrap intervals |
| `results_sensitivity.csv` | Policies at τ = 0.6–0.9 and θ = 0.05, 0.10, 0.20 |
| `tier_confusion.csv` | True tier (rows) against the tier of the predicted intent (columns) |

Both scripts write their output files to the working directory. Copies of every output,
as used in the paper, are in `results/`.

### Column names

| Column | Meaning |
|---|---|
| `automation` | Share of all requests resolved autonomously by the AI |
| `uar_t3` | Unsafe automation: share of true Tier 3 requests resolved autonomously |
| `uar_t23` | Unsafe automation: share of true Tier 2–3 requests resolved autonomously |
| `harm` | Residual harmful misclassification: share of autonomous resolutions that were wrong and had a true tier of 2 or 3 (secondary diagnostic) |
| `over_esc_t1` | Over-escalation: share of true Tier 1 requests not resolved autonomously |
| `under_tier_t3` | Under-tiering: share of true Tier 3 requests routed below Tier 3 |

All values are percentages.

## Tier coding

Each of the 77 Banking77 intents is scored from 1 to 3 on:

- **C – Consequence:** harm if the response or action is wrong
- **U – Uncertainty:** judgement needed about who is asking and what is true
- **I – Irreversibility:** how hard the outcome is to undo

The tier is **T = max(C, U, I)**. Scores are in `intent_coding.csv`; definitions are in
Section II-A of the paper. `Supplementary_S1_Intent_Coding.xlsx` contains the same
coding with columns for independent coders and an automatic agreement calculation.

Intent labels are kept exactly as released in Banking77, including
`Refund_not_showing_up` and `reverted_card_payment?`.

## Optional baselines

These scripts were not used for the results reported in the paper.

- `finetuned_transformer_baseline.py` trains BERT (or RoBERTa) on the same split and
  saves validation and test probability matrices that can replace the TF-IDF
  probabilities in `analysis2.py`. A GPU is recommended.

  ```bash
  pip install transformers torch datasets accelerate
  python finetuned_transformer_baseline.py
  ```

- `llm_zeroshot_baseline.py` labels queries with a large language model's zero-shot
  intent prediction and self-reported confidence. It requires an API key and supports
  P0 and P1 routing only.

## Data and licence

The Banking77 dataset is by PolyAI and is released under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
It is included unchanged in `data/`. If you use it, please cite:

> I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, and I. Vulić, "Efficient intent
> detection with dual sentence encoders," in *Proc. 2nd Workshop on Natural Language
> Processing for Conversational AI*, 2020, pp. 38–45, doi: 10.18653/v1/2020.nlp4convai-1.5.

The code and the tier coding in this repository are released under the MIT licence
(see `LICENSE`).

## Citation

If you use this code or the tier coding, please cite the paper:

```bibtex
@article{dave2026humanai,
  author  = {Nandish Dave},
  title   = {Human--AI Collaborative Customer Support for Digital Asset Platforms: A Trust-Aware Framework for Security, Explainability, and Service Reliability},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

To cite this code and data specifically (release v1.0):

```bibtex
@software{dave2026code,
  author    = {Nandish Dave},
  title     = {risk-tiered-support-routing: Code and data for risk-tiered human--AI customer support},
  version   = {v1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23334097},
  url       = {https://doi.org/10.5281/zenodo.23334097}
}
```
