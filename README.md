# CCA: Canonical Correlation Analysis of Wine Chemistry and Language

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the canonical correlation analysis (CCA) pipeline used to test the "taste bottleneck" hypothesis in wine perception research.

## Hypothesis

A high-dimensional (~100D) chemical sensometabolome interacts with a relatively high-dimensional (~25D) conceptual space of wine language through the mediation of a low-dimensional (~6D) taste space. This bottleneck explains the poor correlation between wine chemistry and wine terminology.

## Data Sources

- **Chemical data**: UCI Wine Quality dataset
  - 1,599 red wine samples
  - 11 physicochemical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
  - Reference: Cortez et al., 2009

- **Linguistic data**: Kaggle Wine Reviews dataset
  - 129,971 wine reviews
  - Filtered to 128,827 reviews containing sensory terminology
  - TF-IDF matrix built using 72-term sensory vocabulary
  - Subsample

## Analysis Pipeline

### 1. Direct CCA
- 11-dimensional chemical space vs. 72-dimensional linguistic space
- 11 canonical components extracted
- Mean canonical correlation: 0.205

### 2. Mediated CCA (6D Bottleneck)
- Both spaces independently reduced to 6 dimensions via PCA
- Simulates the hypothesized low-dimensional taste receptor space
- Mean canonical correlation: 0.054 (73.6% reduction)

### 3. Statistical Validation
- Permutation testing (n = 1,000)
- Direct correlation not statistically significant (p = 0.5)

## Key Results

| Model | Dimensions | Mean Canonical Correlation |
|-------|------------|---------------------------|
| Direct CCA | 11D ↔ 72D | 0.205 |
| Mediated CCA | 6D ↔ 6D | 0.054 |

**Variance preserved in 6D PCA bottleneck:**
- Chemical PCs: 85.5%
- Linguistic PCs: 13.7%

**Conclusion:** The 6D taste space acts as an asymmetric filter preserving chemical structure but discarding most linguistic information thus supporting the taste bottleneck hypothesis.


## Dependencies

- Python 3.x
- NumPy
- SciPy
- scikit-learn (PCA, CCA)
- Pandas
- Matplotlib

## Usage

```python
# Example: Run the direct CCA
python algorithm/cca.py
