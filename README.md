# Hydraulic Condition-Monitoring Dissertation Code

This repository contains the reproducible Python data-processing code and generated analytical outputs used in a dissertation on translating hydraulic condition-monitoring evidence into aircraft landing-gear maintenance-management decisions.

## Scope

The analysis uses the UCI **Condition Monitoring of Hydraulic Systems** dataset as a public hydraulic test-rig benchmark. It does not use operational aircraft data and does not establish aircraft-certified maintenance thresholds.

The code:

- verifies the row and sampling-point structure of the source files;
- reads the five cycle-level condition labels;
- calculates mean, population standard deviation, minimum, maximum and range for 14 physical sensors within each 60-second cycle;
- produces 70 interpretable physical-sensor features per cycle;
- calculates label distributions and condition-specific descriptive statistics;
- performs exploratory eta-squared feature screening;
- creates co-occurrence tables and figures; and
- exports processed data and auditable result tables.

No classifier, remaining-useful-life model, train/test split or fitted machine-learning weights are used.

## Repository contents

```text
hydraulic_data_processing.py   Complete processing and analysis code
requirements.txt               Python dependencies
method_specification.json      Machine-readable method specification
results/                       Generated tables, figures and processed data
```

## Source dataset

Helwig, N., Pignanelli, E. and Schutze, A. (2015), *Condition Monitoring of Hydraulic Systems*, UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CW21

Dataset page: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

Download the source ZIP and place it at either:

```text
condition+monitoring+of+hydraulic+systems.zip
```

or:

```text
ref/condition+monitoring+of+hydraulic+systems.zip
```

The raw 73.1 MB dataset is not duplicated in this repository. The UCI page identifies the dataset as CC BY 4.0.

## Reproducing the analysis

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python hydraulic_data_processing.py
```

Outputs are written to `outputs/hydraulic_data_processing/`.

## Interpretation boundary

Eta-squared is used only as an exploratory effect-size measure. The results are benchmark-level hydraulic evidence rather than classification accuracy, causal evidence or a validated aircraft landing-gear maintenance model.
