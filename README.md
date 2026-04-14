# GW230814-AWaRe-ringdown

Compact workflow for generating GW datasets and using the provided waveform reconstruction artifacts.

## Standard waveform generation

Uses IMRPhenomXPHM (default setup):
- JSON config: `generate_samples/config_files/default.json`
- INI config: `generate_samples/config_files/waveform_params.ini`

```bash
cd generate_samples
python generate_sample.py --config-file default.json
```

## pSEOBNR waveform generation

Uses SEOBNRv5PHM (pSEOBNR setup):
- JSON config: `generate_samples/config_files/default_pSEOBNR.json`
- INI config: `generate_samples/config_files/waveform_params_pSEOBNR.ini`

Basic run:
```bash
cd generate_samples
python generate_sample.py --config-file default_pSEOBNR.json
```

Optional ringdown-deviation controls are available (for example `--domega-22`, `--dtau-22`, or sampled ranges via `--domega-range-modes` / `--dtau-range-modes`).

## Waveform reconstruction model

This repo also includes pretrained temporal-attention reconstruction artifacts:
- Checkpoints: `checkpoints_temporal_attn*` directories (`best_model_*.pt`)
- Evaluation outputs: `test_results_temporal_attn*.pt`
- Analysis outputs: `attention_results*` and `calibration_plots*`

These files are ready for loading in your own PyTorch evaluation/inference script.

## How to run the code

From repository root:

Run training (training code from main branch):
```bash
cd generate_samples
python generate_sample.py --config-file default.json            # standard
python generate_sample.py --config-file default_pSEOBNR.json    # pSEOBNR
```
