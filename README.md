# Archaeological retrieval models

This repository contains scripts to fine-tune a **BGE dense retriever**, train a **BERT backbone for a ColBERT adaptation experiment**, and compare the resulting models with retrieval baselines.

## 1. Installation

Use a Python environment with a compatible PyTorch/CUDA installation if you plan to run the scripts on a GPU.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your machine if the default installation is not appropriate. The provided training scripts use `fp16=True`, and model comparison defaults to CUDA.

## 2. Input files

Run all commands from the repository root. The training scripts expect:

```text
data/
  general_pairs.csv       # general / SMART positive query-document pairs
  dataset_pairs.csv       # synthetic archaeological positive query-document pairs
```

Each training CSV must contain `query` and `document` columns. The additional original metadata columns may be retained.

## 3. Train the BGE model

```bash
python supervised-learn-bge-2-stage.py
```

The script starts from `BAAI/bge-large-en-v1.5`, then trains in two stages:

1. General pairs (`data/general_pairs.csv`)
2. Synthetic archaeological pairs (`data/dataset_pairs.csv`)

The final model is saved to:

```text
bge_results/bge_large_stage2/
```

## 4. Train the backbone for the ColBERT experiment

```bash
python supervised-learn-colbert-transplant-2-stage.py
```

**Important:** This script fine-tunes a Sentence Transformers BERT encoder for **backbone transplantation into ColBERT**. It does **not** train a complete ColBERT model end-to-end.

The script currently starts from `bert-base-uncased` and runs these stages in the opposite order to BGE:

1. Synthetic archaeological pairs (`data/dataset_pairs.csv`)
2. General pairs (`data/general_pairs.csv`)

The final stage model is saved to:

```text
colbert_transplant_two_stage_runs/dataset_then_general/stage2/
```

During comparison, `compare-models.py` loads the pretrained `colbert-ir/colbertv2.0` checkpoint and attempts to load the fine-tuned BERT backbone into it. Check the printed missing/unexpected weight keys when reproducing this adaptation.

## 5. Evaluate the models

Before evaluating, provide the separately distributed evaluation CSV at:

```text
data/evaluation_data.csv
```

Expected columns:

```text
query_id,query,category,doc,relevance
```

The evaluation CSV and trained model checkpoints are **not included** in this source package. Complete the training steps above, or modify the paths in the `models_to_test` dictionary in `compare-models.py` to point to your own checkpoints.

Run:

```bash
python compare-models.py
```

The comparison script evaluates BM25, original and adapted BGE, original and adapted ColBERT, and a cross-encoder baseline. Results are written to:

```text
compare_results/per_query_model_results.csv
compare_results/overall_summary.csv
compare_results/category_summary.csv
compare_results/query_counts_by_category.csv
```

## 6. Reproducibility notes

- Training uses `MultipleNegativesRankingLoss` and validation-based checkpoint selection.
- The scripts currently split train/validation **by row**; documents with multiple query variants may therefore appear in both partitions. Use a document-grouped split for document-disjoint validation.
- Both training scripts define a custom document-unique batch sampler. This does not prevent all semantic false negatives.
- The evaluation script currently excludes BM25 documents with zero/nonpositive scores before ranking. Review this behavior before making strict cross-model comparisons.
- Record the Python version, dependency versions, GPU/CUDA configuration, random seeds, and pretrained model revisions for an exact published reproduction.
- The scripts were inspected, not run through full GPU training and evaluation as part of the package review.
