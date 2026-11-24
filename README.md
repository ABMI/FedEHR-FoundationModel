# AUSOM EHR Foundation Model and Federated Learning

This repository contains the code for a study of how differences between sites affect federated learning of electronic health record (EHR) foundation models (FMs).

A GPT-2 model is pretrained from scratch on patient timelines from an OMOP Common Data Model (CDM) database: the Ajou University School of Medicine (AUSOM) database. Seven virtual sites that share no patients are drawn from the same database, each with a different patient mix. The code then trains and compares three kinds of FM:

- an FM per site;
- one FM trained with FedAvg over all seven sites (FedAvg-7);
- leave-one-site-out FedAvg models over six sites (FedAvg-6), which measure what each site's participation adds.

Every FM is evaluated with frozen representations on 15 downstream tasks, following the EHRSHOT protocol.

## How it works

| Step | Code | What it does |
|---|---|---|
| Base preprocessing | `ausom_ehr.data`, `values`, `tokens`, `tasks` (`scripts/prepare.sh`) | Samples patients and splits them 70/15/15 by patient. Extracts condition, drug, procedure, measurement and observation events. Parses result values, converts units and computes train-only deciles. Builds tokens, visits, index points and labels. |
| Pretraining | `ausom_ehr.tokens`, `train` (`scripts/train_model.sh`) | Builds joint concept + value-state tokens and per-patient sequences, then trains GPT-2 by next-token prediction. |
| Virtual sites | `ausom_ehr.federated site-profile`, `site-disjoint` (`scripts/prepare_site.sh`) | Draws six sites of 300,000 patients each from outside the reference site, with no shared patients. Each site matches one target share (e.g. 80% pediatric); the other axes stay as close to the natural distribution as possible (linear programming over cells). Every site is tokenized with the reference site's rules and vocabulary. |
| FedAvg | `ausom_ehr.federated server`, `client` (`scripts/run_fedavg_disjoint.sh`, `run_fedavg_loso.sh`) | Trains from shared initial weights. One round is one local epoch per site, and weights are averaged by train-token count. Each client keeps its own optimizer state across rounds. |
| Evaluation | `ausom_ehr.evaluate`, `ehrshot_tasks` (`scripts/run_ehrshot_eval.py`, `run_cross_eval.sh`, `run_cross_matrix.py`) | Embeds patients with the frozen FM (last-token hidden state) and fits L2 logistic regression on the evaluation site's train set. C is chosen by validation AUROC, and the test set is scored once. |
| Statistics | `scripts/paired_ehrshot15.py`, `paired_transfer.py`, `scripts/analysis/` | Computes paired differences in test AUROC with patient-level bootstrap (1,000 rounds), and runs subgroup analyses. |

### Sites

| Site | Patient mix |
|---|---|
| A | Random sample of the database (reference site; the base model M0 is trained on it) |
| D0_random | Random sample, same distribution as A |
| D1_pediatric | 80% aged 0–18 |
| D2_er | 80% with an emergency visit |
| D3_cancer | 25% with cancer |
| D4_female | 80% female |
| D5_elderly | 60% aged 65+ |

### Model and training

| Item | Setting |
|---|---|
| Model | GPT-2, 12 layers, 768 dimensions, learned absolute positions (116.5M parameters) |
| Input | The most recent 16,384 tokens; when a record is longer, whole oldest days are dropped |
| Tokens | Concept tokens joined with value states (reference range or train decile, NEG/POS), day-gap tokens, sex and age tokens |
| Optimization | AdamW (β 0.9, 0.95; weight decay 0.1), 65,536 tokens per step |
| Schedule | First epoch: 5% warmup, then cosine decay from 2e-4 to 2e-5. After that: fixed 2e-5, with early stopping on validation loss (patience 3, min delta 0.001) |

### Downstream tasks

| Group | Tasks | Prediction time |
|---|---|---|
| Operational | `los7` (stay of 7 days or more), `readmit30` (readmission within 30 days of discharge), `outpatient90` (admission within 90 days of an outpatient visit) | 23:59 on the index date |
| Lab abnormality | thrombocytopenia, hyperkalemia, hypoglycemia, hyponatremia, anemia | 23:59 on the day before the lab |
| New diagnosis within 365 days | hypertension, type 2 diabetes, hyperlipidemia, heart failure, chronic kidney disease, ischemic stroke, atrial fibrillation | 23:59 on the discharge date |

## Requirements

- Python 3.11 or later (tested with 3.12)
- NVIDIA GPUs. Training and embedding were run on RTX 6000 Ada (48 GB) GPUs, and the code checks for that GPU name.
- An OMOP CDM database exported to Parquet: `<raw_root>/dbo/<TABLE>/batch-*/*.parquet`, with an export marker `_COMPLETE.json` in `<raw_root>` and in each table folder

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[test]"
```

## Configuration

Copy an example config to `configs/local_*.toml` and set `raw_root` to the Parquet export. Output paths are relative to the repository root, where the scripts expect them. Local configs are excluded from version control.

```bash
cp configs/example_base.toml configs/local_base.toml
cp configs/example_m0.toml configs/local_m0.toml
export RAW_ROOT=/path/to/omop_parquet   # used by federated eval and ehrshot_tasks (default: data/omop_parquet)
```

| Config | Purpose |
|---|---|
| `example_base.toml` | Base preprocessing of the reference site A (`outputs/base`) |
| `example_m0.toml` | Base model M0 (`outputs/m0`) |
| `example_m0_seed43.toml` | M0 repeat with seed 43 |
| `example_site_D0_random.toml`, `example_site_D0_random_joint.toml` | One virtual site: base preprocessing and model inputs. Copy once per site. |

## Usage

Every stage runs as `python -m ausom_ehr.<module> <stage> --config CONFIG`. The scripts chain the stages.

```bash
# Base model M0 on the reference site
bash scripts/prepare.sh configs/local_base.toml
GPU_IDS=0 bash scripts/train_model.sh configs/local_m0.toml
bash scripts/run_baselines.sh configs/local_m0.toml

# Virtual sites with no shared patients
python -m ausom_ehr.federated site-profile
python -m ausom_ehr.federated site-make       # D0_random (random site)
python -m ausom_ehr.federated site-disjoint   # D1-D5, disjoint from A, D0 and each other
SITES_ROOT=outputs/sites_disjoint CONFIG_PREFIX=configs/local_disjoint bash scripts/prepare_site.sh D1_pediatric

# Per-site FM (20 epochs)
GPU_IDS=0 EPOCHS=20 bash scripts/train_model.sh configs/local_disjoint_D1_pediatric_joint.toml

# FedAvg over 7 sites, then leave-one-site-out FedAvg over 6 sites
bash scripts/run_fedavg_disjoint.sh
bash scripts/run_fedavg_loso.sh D5_elderly
bash scripts/run_loso_followup.sh D5_elderly

# 12 EHRSHOT tasks, cross-site evaluation and paired comparisons
python -m ausom_ehr.ehrshot_tasks labels --site D1_pediatric   # then inputs; once per site
python -m ausom_ehr.ehrshot_tasks inputs --site D1_pediatric
python scripts/run_ehrshot_eval.py
bash scripts/run_cross_eval.sh D5_elderly "0:A,D0_random 1:D1_pediatric,D2_er"
python scripts/paired_ehrshot15.py
python scripts/paired_transfer.py --fl6 D5_elderly
```

The FedAvg scripts assign each site to a GPU in a table at the top of the script. Edit that table to fit your machine. A rerun of any script skips finished rounds, sites and evaluations.

When preprocessing meets a case that has no handling rule (missing or inconsistent dates, unparsed result values, unit conflicts, days longer than the context), it writes the type and count to an audit file and stops at that stage.

## Outputs

Everything is written under `outputs/`, which is excluded from version control.

| Path | Contents |
|---|---|
| `base/`, `sites_disjoint/<site>/base/` | Base preprocessing of each site: sample, events, value states, vocabulary, labels |
| `m0/`, `sites_disjoint/<site>/joint/` | Model folder (`<model>`) of each site: joint tokens, sequences, task inputs, training and evaluation |
| `<model>/training/metrics.json`, `training/continued/metrics.json` | Training and validation loss per epoch |
| `<model>/evaluation/continued/epoch<N>/results.json` | Test AUROC and AUPRC with bootstrap 95% CIs, plus the selected C, for each task |
| `sites_disjoint/sites_summary.json` | Composition of each virtual site |
| `sites_disjoint/fedavg_7sites/`, `fedavg6_minus_<site>/` | FedAvg weights per round, `metrics.json` (aggregation weights, loss per site and round) |
| `sites_disjoint/eval/`, `eval_ehrshot/` | Evaluation of each FM at each site (`<model>@<site>`) |
| `sites_disjoint/analysis/*.json` | Paired comparisons and subgroup analyses |

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests use synthetic patients and tokens. They cover splitting, event extraction, value states, unit conversion, date ordering and truncation, the index-date cutoff, next-token loss, checkpoint resume and FedAvg averaging.

## Data privacy

This repository contains no patient-level data, model weights or results. `outputs/` and `configs/local_*.toml` are excluded from version control, and the analysis scripts print aggregate numbers only.

## License

To be added.
