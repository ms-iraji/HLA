# Hybrid-based Learning Automata (HLA) Framework

This repository contains the official Python implementation of the UFSM, WLA, and HLA algorithms
as described in the manuscript:

> "Hybrid-based Learning Automata for Feature Selection in Cancer Classification using High-Dimensional Microarray Data"

## Structure
- **ufsm.py** — Algorithm 1 (UFSM: Unsupervised Filter-based Similarity Measure)
- **wla.py** — Algorithm 2 (Wrapper-based Learning Automata)
- **hla.py** — Algorithm 3 (Hybrid-based Learning Automata)
- **demo_main.ipynb** — Example notebook reproducing results using a sample dataset

## Data
All datasets used in the manuscript (Colon, CNS, GLI-85, SMK-CAN-187, Leukemia)
are publicly available from the Gene Expression Omnibus (GEO) and UCI repositories.
No new data were collected involving humans or animals.

## Usage in Google Colab
```python
!git clone https://github.com/ms-iraji/HLA_Code.git
cd HLA_Code
!pip install -r requirements.txt
```

Run the notebook:
```python
!jupyter nbconvert --to notebook --execute demo_main.ipynb
```

