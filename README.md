# GW230814-AWaRe-ringdown

Compact workflow for waveform generation and temporal-attention waveform reconstruction.

## Waveform generation: pSEOBNR

pSEOBNR generation uses SEOBNRv5PHM with:
- JSON: `generate_samples/config_files/default_pSEOBNR.json`
- INI: `generate_samples/config_files/waveform_params_pSEOBNR.ini`

Basic run:
```bash
cd generate_samples
python generate_sample.py --config-file default_pSEOBNR.json
```

Useful pSEOBNR options:
```bash
# Multi-detector + multiple noise realizations
python generate_sample.py --config-file default_pSEOBNR.json --detectors H1 L1 --n-noise-realizations 2

# Fixed ringdown deviations for selected modes
python generate_sample.py --config-file default_pSEOBNR.json --domega-22 0.05 --dtau-22 -0.03 --domega-33 0.02 --dtau-33 0.01

# Sample deviations from INI ranges for selected modes
python generate_sample.py --config-file default_pSEOBNR.json --domega-range-modes 22 33 --dtau-range-modes 22 33
```

## Waveform generation: everything else

Standard/default generation (IMRPhenomXPHM):
- JSON: `generate_samples/config_files/default.json`
- INI: `generate_samples/config_files/waveform_params.ini`

```bash
cd generate_samples
python generate_sample.py --config-file default.json
```

Other available configs:
- `default_NRSur.json` + `waveform_params_NRSur.ini` (NRSur7dq4)
- `default_IMBH.json` + `waveform_params_IMBH.ini` (IMBH-focused setup)

## Waveform reconstruction model (brief)

The reconstruction model is a temporal-attention network that learns to map noisy detector time-series to cleaned/reconstructed GW waveforms for ringdown-focused analysis.

Repository artifacts:
- Best checkpoints: `checkpoints_temporal_attn*/best_model_*.pt`
- Test outputs: `test_results_temporal_attn*.pt`
- Attention/calibration summaries: `attention_results*`, `calibration_plots*`

Run training (training code from main branch):
```bash
python train_temporal_attn.py
```
