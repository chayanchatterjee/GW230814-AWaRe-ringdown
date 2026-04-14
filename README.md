# GW230814-AWaRe-ringdown

This repository contains tooling for generating gravitational-wave (GW) training/evaluation datasets (signal injections + detector noise), with a focus on ringdown-aware studies and waveform systematics.

## Repository layout

- `generate_samples/`: Main sample-generation package.
  - `generate_sample.py`: Main entry point; reads JSON+INI config files and writes HDF5 sample files.
  - `config_files/`: Ready-to-run JSON/INI configurations for different waveform families.
  - `utils/`: Core utilities for waveform drawing, noise timeline sampling, HDF I/O, and sample assembly.
  - `Generate_pSEOBNR.ipynb`: Notebook workflow for pSEOBNR experiments.
  - `create_real_events_file.py`, `download_gwosc_data.py`: Utilities for event-data preparation.
- `checkpoints_temporal_attn*/`: Saved model checkpoints.
- `attention_results*/`, `calibration_plots*/`: Post-training analysis and calibration artifacts.

## Sample generation overview

The generation pipeline is controlled by:

1. A **JSON config** (for run-level settings like number of samples, output names, noise source paths, process count).
2. An **INI config** (for waveform approximant, static sampling parameters, and prior distributions).

`generate_sample.py` combines both, samples waveform parameters, draws noise intervals, generates injections/noise-only examples, and writes outputs to HDF5.

---

## Quick start

From repository root:

```bash
cd generate_samples
python generate_sample.py --config-file default.json
```

> Note: `generate_sample.py` resolves config paths relative to `generate_samples/config_files/`, so run commands from `generate_samples/` (or adjust paths in code/config accordingly).

---

## Commands for pSEOBNR waveform generation

The pSEOBNR setup in this repo uses the SEOBNRv5PHM approximant via:

- JSON: `config_files/default_pSEOBNR.json`
- INI: `config_files/waveform_params_pSEOBNR.ini`

### Standard pSEOBNR run

```bash
cd generate_samples
python generate_sample.py --config-file default_pSEOBNR.json
```

### pSEOBNR run with detector selection and multiple noise realizations

```bash
cd generate_samples
python generate_sample.py \
  --config-file default_pSEOBNR.json \
  --detectors H1 L1 \
  --n-noise-realizations 2
```

### pSEOBNR run with mode-wise ringdown deviations

```bash
cd generate_samples
python generate_sample.py \
  --config-file default_pSEOBNR.json \
  --domega-22 0.05 --dtau-22 -0.03 \
  --domega-33 0.02 --dtau-33 0.01
```

### pSEOBNR run using sampled deviation ranges from INI

`waveform_params_pSEOBNR.ini` defines `domega_range` and `dtau_range`. Apply sampled values to selected modes:

```bash
cd generate_samples
python generate_sample.py \
  --config-file default_pSEOBNR.json \
  --domega-range-modes 22 33 \
  --dtau-range-modes 22 33
```

---

## Commands for standard waveform generation

### 1) IMRPhenomXPHM (default standard configuration)

- JSON: `config_files/default.json`
- INI: `config_files/waveform_params.ini`

```bash
cd generate_samples
python generate_sample.py --config-file default.json
```

### 2) NRSur7dq4 waveform generation

- JSON: `config_files/default_NRSur.json`
- INI: `config_files/waveform_params_NRSur.ini`

```bash
cd generate_samples
python generate_sample.py --config-file default_NRSur.json
```

### 3) IMBH-oriented generation

- JSON: `config_files/default_IMBH.json`
- INI: `config_files/waveform_params_IMBH.ini`

```bash
cd generate_samples
python generate_sample.py --config-file default_IMBH.json
```

---

## Frequently used runtime options

`generate_sample.py` supports additional runtime controls, for example:

- `--detectors H1 L1` (or single detector like `--detectors H1`)
- `--n-noise-realizations N`
- `--negative-latency SECONDS`
- `--add_glitches_noise <glitch_name>`
- `--add_glitches_injection <glitch_name>`
- mode-wise deviation flags (e.g. `--domega-22`, `--dtau-22`, etc.)

Example:

```bash
cd generate_samples
python generate_sample.py \
  --config-file default.json \
  --detectors H1 L1 \
  --n-noise-realizations 1 \
  --negative-latency 0
```

## Output

Generated sample files are written as HDF5 using names defined in the selected JSON config, e.g.:

- `output_file_name`
- `snr_output_file_name`
- `template_output_file_name`

Adjust these fields in `generate_samples/config_files/*.json` before running large production jobs.
