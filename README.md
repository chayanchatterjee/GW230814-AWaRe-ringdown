# GW230814-AWaRe-ringdown

Compact workflow for generating GW datasets and using the provided waveform reconstruction artifacts.

## Waveform generation (everything except pSEOBNR)

Uses the default waveform setup (IMRPhenomXPHM):
- JSON config: `generate_samples/config_files/default.json`
- INI config: `generate_samples/config_files/waveform_params.ini`

```bash
cd generate_samples
python generate_sample.py --config-file default.json
```

## Waveform generation (pSEOBNR)

Uses SEOBNRv5PHM (pSEOBNR setup):
- JSON config: `generate_samples/config_files/default_pSEOBNR.json`
- INI config: `generate_samples/config_files/waveform_params_pSEOBNR.ini`

Basic run from the `generate_samples` directory:
```bash
cd generate_samples
python generate_sample.py --config-file default_pSEOBNR.json
```

You can also control ringdown deviations during generation, for example:
- direct values: `--domega-22`, `--dtau-22`
- sampled ranges (multiple modes): `--domega-range-modes`, `--dtau-range-modes`

This is useful when you want to produce datasets with controlled perturbations around the baseline pSEOBNR signal.

## Waveform reconstruction model

This repo also includes pretrained temporal-attention reconstruction artifacts:
- Checkpoints: `checkpoints_temporal_attn*` directories (`best_model_*.pt`)
- Evaluation outputs: `test_results_temporal_attn*.pt`
- Analysis outputs: `attention_results*` and `calibration_plots*`

The reconstruction model is a 1D temporal-attention U-Net with transformer-style blocks:
- **Backbone:** multiscale encoder-decoder (U-Net) over 1D time-series signals.
- **Core blocks:** each level uses pre-norm transformer blocks with temporal multi-head self-attention + feed-forward layers.
- **Position handling:** rotary positional encoding (RoPE) is applied in attention to preserve relative temporal structure.
- **Output head:** predicts two channels per time step (`mu` and `logvar`) for mean reconstruction plus uncertainty-aware training.
- **Interpretability:** attention maps can be extracted from encoder/decoder blocks for post-hoc analysis (e.g., time-localization studies).

During training, masked contiguous segments are used as a calibration signal so the model learns reconstruction and uncertainty together, rather than only point estimates.

These files are ready for loading in your own PyTorch evaluation/inference script.

## Training command

From repository root:

```bash
python train_recons_temporal_atten.py
```
