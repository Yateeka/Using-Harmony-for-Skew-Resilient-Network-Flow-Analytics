# Harmony-Based Skew-Resilient Network Flow Analytics

This project implements a Harmony-inspired distributed vector search system for real-time network flow analytics under skewed and bursty traffic conditions.

We adapt the Harmony architecture to the CIC-IDS2017 dataset, transforming network flow features into vector embeddings and evaluating different partitioning strategies.

---

## Key Features

- Network flow to vector embedding pipeline  
- Evaluation under uniform and skewed workloads  
- Multiple system designs:
  - Vector-based partitioning
  - Dimension-based partitioning
  - Hybrid Harmony-style execution
  - Full Harmony pipeline (algorithm-based)
- Metrics:
  - Latency (mean, P95, P99)
  - Throughput (QPS)
  - Load imbalance
  - Pruning efficiency

---

## Project Structure

```
erp-harmony/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── preprocess_cicids2017.py
│   ├── vector_baseline.py
│   ├── dimension_baseline.py
│   ├── harmony_hybrid.py
│   └── harmony_pipeline.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Setup

Download CIC-IDS2017 dataset (via Kaggle):

```bash
kaggle datasets download -d chethuhn/network-intrusion-dataset -p data/raw
unzip data/raw/network-intrusion-dataset.zip -d data/raw/cicids2017
```

Make sure CSV files are inside:

```
data/raw/cicids2017/
```

---

## Step 1: Preprocessing

```bash
python scripts/preprocess_cicids2017.py
```

### Output

```
data/processed/
├── base_vectors.npy
├── query_vectors.npy
├── queries_uniform.npy
├── queries_skewed.npy
├── base_labels.csv
├── query_labels.csv
```

---

## Step 2: Run Baselines

### Vector Baseline

```bash
python scripts/vector_baseline.py --query-set uniform
python scripts/vector_baseline.py --query-set skewed
```

### Dimension Baseline

```bash
python scripts/dimension_baseline.py --query-set uniform
python scripts/dimension_baseline.py --query-set skewed
```

---

## Step 3: Hybrid Systems

### Hybrid (Vector + Dimension)

```bash
python scripts/harmony_hybrid.py --query-set uniform
python scripts/harmony_hybrid.py --query-set skewed
```

### Full Harmony Pipeline (Recommended)

```bash
python scripts/harmony_pipeline.py --query-set uniform
python scripts/harmony_pipeline.py --query-set skewed
```

---

## Metrics

Each script reports:

- Mean latency  
- P95 / P99 latency  
- Throughput (QPS)  
- Load imbalance ratio  
- Pruning statistics  

---

## Concepts

- Vector partitioning  
- Dimension partitioning  
- Hybrid execution  
- Early pruning  
- Skewed workload simulation  
- Top-k similarity search  

---

## Notes

- This is a research prototype  
- The original Harmony system is implemented in C++ and is not publicly available  
- This project reproduces the core algorithmic ideas in Python  

---

## Authors

- Yateeka Goyal  
- Apu Kumar Chakroborti  

---

## Status

- Data pipeline complete  
- Baselines implemented  
- Harmony pipeline implemented  
- Evaluation in progress  

---

## License

Academic use only
