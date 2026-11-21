# MOF-flexibility

Data and code for **"Mining the Experimental Literature Reveals the Flexibility Landscape of Metal–Organic Frameworks"**.

The pipeline extracts stimulus-resolved flexibility labels (solvent / temperature / gas adsorption–desorption)
for the 8,804 CoRE MOF 2024 structures from their original publications with an LLM, verifies every evidence
sentence against the source text, trains a random forest on structure-only descriptors under a
leakage-controlled (material-grouped) evaluation, and extends gas-axis labels to unlabeled structures.

## Layout (numbered in pipeline order)

```
data/                       released data (see below)
examples/                   one open-access example paper (CC BY 4.0) + its CIF for an end-to-end test
common/data.py              loader that serves the released files to every script
01_extraction/   extract_labels.py            LLM label extraction (prompt embedded verbatim)
                 ground_evidence.py           evidence-sentence grounding + 47 claim keywords (loose)
                 ground_keywords_strict.py    strict keyword criterion
02_grouping/     compute_mofkey.py            MOFkey via mofid
                 compute_topology_descriptors.py   CrystalNets fingerprint (vertices, edges, td10)
                 make_material_groups.py      -> data/master_groups.csv (material / chemistry / doi / combined)
03_mechanical/   compute_elastic_tensors.py   MLIP (orb-v3 + D3) elastic tensors, sharded
                 apply_qc.py                  shard merge + acceptance criteria (7,828 of 8,802)
                 validation/                  MLIP vs DFT on MOF-5, UiO-66, MIL-53 lp/np (SI S5)
04_modeling/     make_folds.py                frozen 5-fold assignments for the split ladder (run first)
                 train_baselines.py           RF / XGB / GBR at default settings, 5 split levels
                 tune_rf_nested_cv.py         nested-CV RF tuning (Table 2, S9, S11)
                 refit_rf_final.py            final RF on all seeds (941/12/8/sqrt, balanced)
                 cgcnn/                       CGCNN baseline (S10)
05_evaluation/   paired_bootstrap.py, bootstrap_to_table.py   paired bootstrap ΔAUC (S8)
                 interval_precision.py        interval-wise precision, fixed hyperparameters (Table 3, S15c)
                 shap_oof.py                  out-of-fold TreeSHAP (Figure 4, S12)
06_label_extension/ predict_unlabeled.py      P(yes) and confident labels for the 5,350 unlabeled CIFs (Figure 5)
                 ls_compute_rev2.py           label-spreading cross-check with the corrected labels: pooled gate grid
                                              over 25 fold seeds, 10 check seeds, comparison with the final RF (Figure 6, S13)
                 label_spreading.py, ls_gate_validation.py   earlier label-spreading scripts (superseded by ls_compute_rev2.py)
07_external_check/  compute_geometry.py, compute_racs.py, compute_elastic.py, predict.py   NU-2002 (S15)
08_figures/      plot_rf_vs_ls.py             Figure 6
results/         reference outputs used in the paper
```

## Data (`data/`)

| file | content |
|---|---|
| `LLM_Extraction.xlsx` | 8,804 structures: identifiers, DOI, per-stimulus labels (yes/no/unknown/none), guest-resolved labels, evidence sentences, reasoning |
| `descriptors.csv` | 8,804 × 191 descriptors: 10 geometric (Zeo++ values from CoRE MOF 2024), 176 RACs, 5 mechanical (orb-v3 elastic tensor) + QC fields; `feature_complete` marks the 7,188 modeling structures |
| `labels/labels_final.csv` | labels used for training (8,804 CIFs × solvent / temperature / gas): the extracted labels with the corrections in `labels/label_changes.csv` |
| `labels/label_changes.csv` | the six label corrections (CIF, axis, old value, new value, reason) |
| `labels/hp_scan.csv` | search for gas Yes labels that come from high-pressure experiments, and how each was handled |
| `master_groups.csv` | material / chemistry / DOI / combined group identifiers for the split ladder (412 CIFs without a MOFkey grouped by `fallback_groups.csv`) |
| `fallback_groups.csv` | fallback material groups (DOI + metal) for the CIFs without a MOFkey |
| `topology.csv` | CrystalNets descriptors used for the fallback material identifier |
| `NU2002_ordered.cif` | ordered, hydrogen-completed model of NU-2002 (CCDC 2234121) |

## Results (`results/`)

| path | content |
|---|---|
| `folds/<level>/folds/<axis>_folds.csv` | frozen 5-fold assignments (cif_name, y, fold) for the five split levels, with the corrected labels; `folds/moves.csv` lists the CIFs moved after the regrouping |
| `cv/predictions_all.csv` | out-of-fold scores of every model (RF default, RF tuned, RF geometric-only, RF RAC-only, GBR, XGBoost, CGCNN), 5 splits × 3 axes |
| `cv/rf_tuned_params.csv`, `cv/rf_tuned_inner_cv.csv` | nested-CV RF: selected hyperparameters and inner-CV scores per fold |
| `final_rf/` | final model (default RF, mean of 10 forests): `oof_<axis>.csv`, `probabilities_<axis>.csv` / `probabilities_gas_final.csv` (P(yes) of the unlabeled CIFs), `final_numbers.json`, `headline_extension.json`, `calibration_material_level.json`; temperature files with COZMIQ02 = yes (1,177 seeds); `final_numbers_v2.json` = AUC ladder, paired differences and S8 recomputed with these and the retrained comparison models (supersedes those sections of `final_numbers.json`) |
| `label_spreading/` | label-spreading cross-check (SI S13): gate grids (`grid_*.csv`), per-target results (`ls_gas_targets_rev2.csv`), `compute.json`; `ls_rows_7188.csv` is the row order and the labels as extracted |
| `evidence_keyword_grounding_full.csv` | evidence-sentence grounding per CIF and axis (the passages of the source articles are not included) |

Final model: `RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features='sqrt', bootstrap=False, class_weight='balanced')`,
10 forests with random_state 20260724–20260733, P(yes) = mean of the 10; confident intervals No P ≤ 0.10, Yes P > 0.85
(0.80 < P ≤ 0.85: second tier). Training rows in `descriptors.csv` order, scikit-learn 1.9.x.
Label spreading (S13): No gate k = 30, α = 0.2; Yes gate k = 75, α = 0.2; support ≥ 3 seed materials; 25 fold seeds.

Not redistributed: full texts of the source publications (see `examples/` for one CC BY paper),
CSD-derived chemical names, and the CoRE MOF 2024 CIFs (available from the CoRE MOF repository).

## Reproducing the paper

Run each script from its own directory. All scripts read `data/` through `common/data.py`.

| step | command |
|---|---|
| folds | `04_modeling/`: `python make_folds.py` → `results/folds/` (reproduces the paper folds exactly) |
| Table 2, S9, S11 | `04_modeling/`: `python train_baselines.py`; `python tune_rf_nested_cv.py` (`--levels 2_mofkey` for the material split only) |
| S8 | `05_evaluation/`: `python paired_bootstrap.py`; `python bootstrap_to_table.py` |
| Figure 4, S12 | `05_evaluation/`: `python shap_oof.py --levels 2_mofkey --params ../results/rf_best_params.csv --top 10` |
| Table 3, S15c | `05_evaluation/`: `python interval_precision.py --folds ../results/oof_predictions_nested.csv` |
| Figure 5 (extended labels) | `04_modeling/`: `python refit_rf_final.py --out_dir ../results/rf_final`; `06_label_extension/`: `python predict_unlabeled.py --out_dir ../results/rf_final` |
| Figure 6, S13 | `06_label_extension/`: `python ls_compute_rev2.py` → `results/label_spreading/` (about 1 min; reproduces the released files exactly with scikit-learn 1.9.0 — the fold-seed splits of `StratifiedGroupKFold` differ in 1.7) |
| S15a/b | `07_external_check/`: `compute_geometry.py`, `compute_racs.py`, `compute_elastic.py`, `predict.py` |
| S5, S6 | `03_mechanical/validation/`, `03_mechanical/apply_qc.py` |

Exact reproduction of tree-based numbers requires the pinned versions in `environment.yml`.

## Environment

See `environment.yml`. Elastic tensors additionally require `orb-models`, `torch-dftd`, `matcalc` and a GPU;
CrystalNets requires Julia (`CrystalNets.jl` via `juliacall`); MOFkey requires `mofid`; extraction requires
`chromadb`, `openai` and Marker-converted full texts.

## Citation

[paper reference] · data/code archive: Zenodo DOI [to be added] · code commit [to be added]
