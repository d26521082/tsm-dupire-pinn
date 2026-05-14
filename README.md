# TSM Dupire-PINN: Local-Volatility Calibration from Sparse Cross-Sectional Equity Options

Code and data accompanying:

> Chen-Ting Lin and Chin-Lung Chou, *Recovering Nonlinear Local-Volatility Structure from Cross-Sectionally Sparse Equity Options via Physics-Informed Neural Networks*. Submitted to *Chaos, Solitons & Fractals*.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20177970.svg)](https://doi.org/10.5281/zenodo.20177970)

---

## What is in this repository

A complete, reproducible pipeline for calibrating the dividend-aware Dupire forward equation against a single-day cross-section of Taiwan Semiconductor (TSM, NYSE) listed equity options, using a physics-informed neural network (PINN). The pipeline includes the data audit, preprocessing, PINN training, three baseline calibrations (slice-fit SVI, cubic-spline interpolation, and a generalised Andreasen--Huge one-step finite-difference scheme), leave-one-maturity-out cross-validation with fold-respecting statistics, PDE-prior ablation, and surface-robustness checks.

## Data

`features_all.csv` — 259 TSM option observations at a single market close (single underlying, single trading day). Columns: `S, K, T, r, sigma, moneyness, market_price, optionType, moneyness_category, expirationDate`. The `sigma` column is a single constant value (the broker's flat IV); per-option implied volatilities are recomputed in `analysis/data_prep.py` after dividend extraction and OTM-only unification.

## Repository structure

```
.
├── features_all.csv                  raw option chain
├── requirements.txt                  pinned Python dependencies
├── analysis/
│   ├── data_audit.py                 data-quality audit (arbitrage, parity, IV)
│   ├── data_prep.py                  filter + implied-q + OTM-only + IV inversion → data/clean.csv
│   ├── train_pinn.py                 Dupire-PINN training (two equinox MLPs, JAX)
│   ├── baselines.py                  SVI + cubic-spline baselines
│   ├── baseline_ah.py                Andreasen-Huge (dividend-extended)
│   ├── cv_evaluate.py                LOMO + LOSO CV, fold-bootstrap, fold-level paired tests
│   ├── ablation.py                   PDE-prior ablation (4 variants × LOMO)
│   ├── sigma_analysis.py             σ_loc surface + per-maturity wing exponents
│   ├── sigma_sensitivity.py          σ_ref sensitivity sweep
│   ├── sigma_robustness.py           T-adaptive + window sensitivity + data bootstrap
│   ├── otm_filter_ablation.py        OTM-only vs all-strike-call comparison
│   ├── american_premium.py           binomial-tree American early-exercise premium
│   ├── iv_vs_lv_consistency.py       σ_iv vs σ_loc cross-comparison at data points
│   └── build_results_tex.py          aggregates all JSON outputs into results.tex + highlights.txt
```

## Environment

The pipeline targets Python 3.11. Dependencies are pinned in `requirements.txt`. JAX is the heaviest dependency (used for PINN training and autodiff).

```bash
conda create -n thesis-pinn python=3.11 -y
conda activate thesis-pinn
pip install -r requirements.txt
```

## Reproduce the paper's numerical results

Run scripts in the following order. All outputs (cleaned data, JSON metrics, figures) are written under `data/` and `figures/`.

```bash
# 1. Data audit (no output files; prints quality report)
python analysis/data_audit.py

# 2. Preprocess: filter, extract q from parity, OTM-only unify, invert IV
python analysis/data_prep.py
# → data/clean.csv

# 3. Train PINN (full data) and produce the in-sample surface
python analysis/train_pinn.py
# → data/pinn_model.eqx, figures/pinn/*.png

# 4. SVI + cubic-spline baselines
python analysis/baselines.py
# → data/svi_params.json, data/baseline_metrics.json

# 5. Andreasen-Huge baseline
python analysis/baseline_ah.py

# 6. Leave-one-maturity-out CV across all four models
python analysis/cv_evaluate.py
# → data/cv_summary.json, figures/cv/*.png

# 7. PDE-prior ablation
python analysis/ablation.py
# → data/ablation_summary.json

# 8. σ_loc surface + wing-exponent analysis
python analysis/sigma_analysis.py
# → data/sigma_analysis.json, figures/sigma/*.png

# 9. σ_ref sensitivity
python analysis/sigma_sensitivity.py

# 10. T-adaptive + bootstrap robustness
python analysis/sigma_robustness.py
# → data/sigma_robustness.json, figures/robustness/*.png

# 11. OTM-only vs all-strike ablation
python analysis/otm_filter_ablation.py

# 12. American early-exercise premium estimate
python analysis/american_premium.py

# 13. σ_iv vs σ_loc consistency plot
python analysis/iv_vs_lv_consistency.py

# 14. Aggregate everything into results.tex (consumed by the manuscript)
python analysis/build_results_tex.py
```

Total compute time on a recent multi-core CPU (no GPU required): approximately 60--90 minutes, dominated by `cv_evaluate.py` (9 LOMO folds × 4 baselines) and `ablation.py` (4 variants × 9 LOMO folds).

## Reproducibility notes

All training uses fixed random seeds (declared in each script). The PINN ensemble in `sigma_analysis.py` averages over five seeds (`42, 7, 123, 2024, 99`). The data-bootstrap analysis in `sigma_robustness.py` uses ten resamples with seeds `1042, 1043, ..., 1051`.

The Andreasen-Huge implementation generalises the Risk Magazine 2011 scheme to include `(r-q) K ∂_K` drift and `-q C` discount terms, so it is directly comparable to the dividend-aware Dupire PDE used in the PINN.

## Citation

If you use this code or data, please cite the paper (BibTeX entry will be added on publication) and the Zenodo archive:

```
@dataset{lin_chou_tsm_dupire_pinn,
  author       = {Lin, Chen-Ting and Chou, Chin-Lung},
  title        = {TSM Dupire-PINN: Code and Data},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20177970},
  url          = {https://github.com/d26521082/tsm-dupire-pinn}
}
```

## License

- Source code (everything under `analysis/`) is released under the **MIT License** (see `LICENSE-CODE`).
- The raw option-chain data and derived datasets are released under **CC-BY-4.0** (see `LICENSE-DATA`).

## Contact

Chen-Ting Lin — Department of Economics, National Taiwan University — d26521082@gmail.com

Corresponding author: Chin-Lung Chou — Management Undergraduate Program, National Taiwan University of Science and Technology.
