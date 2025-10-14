# virgo-data-scripts

Small, focused repository for GW/Virgo data utilities and scripts.

## Highlights
- Simple layout with two main folders:
  - `scripts/` — executable utilities (Python, shell, etc.)
  - `channels/` — channel lists and related metadata
- Batteries-included examples and docs kept close to the code.
- Everything else will evolve as the project grows.

## Layout
```bash
virgo-data-scripts/ \
├── channels/ \
│ ├── README.md \
│ ├── all_channels.txt \
│ ├── aux_channels.txt \
│ ├── Sa_channels.txt \
│ └── Sc_channels.txt \
├── scripts/ \
│ └── gwf_to_h5_batch.py \
├── .gitignore \
├── LICENSE \
├── README.md \
├── requirements.txt \
└── .github/ \
└── workflows/ \
└── ci.yml \
```

## Quickstart

```bash
# 1) (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
python -m pip install -U pip
pip install -r requirements.txt

# 3) Run the example script
python scripts/gwf_to_h5_batch.py \\
  --ffl raw \\
  --start "2025-08-14 00:00:00" \\
  --duration 1h \\
  --channels-file channels/aux_channels.txt \\
  --out dataset/aux_1h/dataset.h5 \\
  --nproc 4 --scan-limit 3 --verbose 1
```