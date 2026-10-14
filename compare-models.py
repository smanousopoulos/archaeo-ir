import os
import torch
import json
import numpy as np
import pandas as pd
import random
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

RESULTS_OUTPUT = "compare_results"

def shuffle_docs(docs, seed=42):
    rng = random.Random(seed)
    docs = docs.copy()
    rng.shuffle(docs)
    return docs

def load_eval_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and "queries" in data:
        raw_queries = data["queries"]
    elif isinstance(data, dict):
        raw_queries = [data]
    elif isinstance(data, list):
        raw_queries = data
    else:
        raise ValueError("JSON must be a query dict, a list of query dicts, or a dict with a 'queries' key.")

    queries = []
    for i, item in enumerate(raw_queries):
        raw_gold = item.get("gold_standard", {}) or {}
        if not isinstance(raw_gold, dict):
            raise ValueError(f"gold_standard for item {i} is not a dict")

        gold_standard = {
            str(doc): int(float(rel))
            for doc, rel in raw_gold.items()
            if doc is not None and str(doc).strip()
        }

        queries.append({
            "query_id": item.get("query_id", f"q_{i}"),
            "text": str(item.get("text", "") or "").strip(),
            "gold_standard": gold_standard
        })

    return {"queries": queries}

def load_eval_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"query_id", "query", "category", "doc", "relevance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    queries = []
    grouped = df.groupby("query_id", sort=False)

    for query_id, group in grouped:
        group = group.copy()

        group["relevance"] = (
            pd.to_numeric(group["relevance"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        group = group[group["doc"].notna() & group["doc"].astype(str).str.strip().ne("")]

        query_text = str(group["query"].iloc[0]).strip()
        category = str(group["category"].iloc[0]).strip() if pd.notna(group["category"].iloc[0]) else "unknown"

        gold_standard = dict(zip(group["doc"], group["relevance"]))

        queries.append({
            "query_id": query_id,
            "text": query_text,
            "category": category,
            "gold_standard": gold_standard,
        })

    return {"queries": queries}


def get_metrics(gold_dict, pred_scores, k=10):
    ranked = sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True)
    top_k = ranked[:k]

    rel_ids = {d for d, s in gold_dict.items() if s > 0}
    R = len(rel_ids)
    top_r = ranked[:R] if R > 0 else []

    hits_k = [d for d in top_k if d in rel_ids]
    hits_r = [d for d in top_r if d in rel_ids]

    pk = len(hits_k) / k if k > 0 else 0
    rk = len(hits_k) / R if R > 0 else 0
    f1k = (2 * pk * rk) / (pk + rk) if (pk + rk) > 0 else 0

    # nDCG@k
    dcg_k = sum(
        (2 ** gold_dict.get(d, 0) - 1) / np.log2(i + 2)
        for i, d in enumerate(top_k)
    )
    ideal_k = sorted(gold_dict.values(), reverse=True)[:k]
    idcg_k = sum(
        (2 ** v - 1) / np.log2(i + 2)
        for i, v in enumerate(ideal_k)
    )
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0

    # nDCG@3
    top_3 = ranked[:3]
    dcg_3 = sum(
        (2 ** gold_dict.get(d, 0) - 1) / np.log2(i + 2)
        for i, d in enumerate(top_3)
    )
    ideal_3 = sorted(gold_dict.values(), reverse=True)[:3]
    idcg_3 = sum(
        (2 ** v - 1) / np.log2(i + 2)
        for i, v in enumerate(ideal_3)
    )
    ndcg_3 = dcg_3 / idcg_3 if idcg_3 > 0 else 0

    # Recall@R
    recall_r = len(hits_r) / R if R > 0 else 0

    return {
        f"P@{k}": pk,
        f"R@{k}": rk,
        f"F1@{k}": f1k,
        f"nDCG@{k}": ndcg_k,
        #f"nDCG@3": ndcg_3,
        #"R@R": recall_r,
        "NumRelevant": R
    }, ranked

class ColBERTScorer:
    def __init__(self, model_path, device="cuda"):
        config = ColBERTConfig(checkpoint=model_path)
        self.ckpt = Checkpoint(model_path, colbert_config=config)

        self.device = device
        self.ckpt.model.to(device)
        self.ckpt.model.eval()

    def score(self, query, docs):
        with torch.no_grad():
            Q = self.ckpt.queryFromText([query]).to(self.device).float()
            D = self.ckpt.docFromText(docs).to(self.device).float()

            scores = torch.matmul(Q, D.permute(0, 2, 1))
            scores = scores.max(dim=2).values.sum(dim=1)

        return dict(zip(docs, scores.cpu().tolist()))

    def cleanup(self):
        del self.ckpt

class BGEScorer:
    def __init__(self, model_path, device="cuda"):
        self.model = SentenceTransformer(model_path, device=device)
        self.device = device

        # BGE instruction
        self.instruction = "Represent this sentence for searching relevant passages: "

    def score(self, query, docs):
        full_query = self.instruction + query

        q_emb = self.model.encode(full_query, convert_to_tensor=True)
        d_embs = self.model.encode(docs, convert_to_tensor=True)

        scores = util.cos_sim(q_emb, d_embs)[0]

        return dict(zip(docs, scores.cpu().tolist()))

    def cleanup(self):
        del self.model

class ColBERTTransplantScorer:
    def __init__(self, backbone_path, device="cuda"):
        config = ColBERTConfig(checkpoint="colbert-ir/colbertv2.0")
        self.ckpt = Checkpoint(config.checkpoint, colbert_config=config)

        self.device = device

        new_brain = AutoModel.from_pretrained(backbone_path)
        result = self.ckpt.model.bert.load_state_dict(new_brain.state_dict(), strict=False)

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Missing keys: {result.missing_keys}\n"
                f"Unexpected keys: {result.unexpected_keys}"
            )

        self.ckpt.model.to(device)
        self.ckpt.model.eval()

    def score(self, query, docs):
        with torch.no_grad():
            Q = self.ckpt.queryFromText([query]).to(self.device).float()
            D = self.ckpt.docFromText(docs).to(self.device).float()

            scores = torch.matmul(Q, D.permute(0, 2, 1))
            scores = scores.max(dim=2).values.sum(dim=1)

        return dict(zip(docs, scores.cpu().tolist()))

    def cleanup(self):
        del self.ckpt

class CrossEncoderScorer:
    def __init__(self, model_path, device="cuda", max_length=512):
        self.model = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length
        )
        self.device = device

    def score(self, query, docs):
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Convert numpy / tensor-ish values to plain floats
        scores = [float(s) for s in scores]
        return dict(zip(docs, scores))

    def cleanup(self):
        del self.model

def simple_tokenize(text):
    return str(text).lower().split()


class BM25Scorer:
    def __init__(self):
        pass

    def score(self, query, docs):
        tokenized_docs = [simple_tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = simple_tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        return dict(zip(docs, scores.tolist()))

    def cleanup(self):
        pass

def build_scorers(models_to_test, device="cuda"):
    scorers = {}

    for name, (path, m_type) in models_to_test.items():
        print(f"Loading model: {name}")

        if m_type == "colbert":
            scorers[name] = ColBERTScorer(path, device=device)

        elif m_type == "dense":
            scorers[name] = BGEScorer(path, device=device)

        elif m_type == "transplant":
            scorers[name] = ColBERTTransplantScorer(path, device=device)

        elif m_type == "cross_encoder":
            scorers[name] = CrossEncoderScorer(path, device=device)

        elif m_type == "bm25":
            scorers[name] = BM25Scorer()

        else:
            raise ValueError(f"Unknown model type: {m_type}")

    return scorers

CATEGORY_ORDER = [
    "1 one_concept",
    "1.5 one_concept_modifiers",
    "2 two_concept",
    "2.5 two_concept_modifiers",
    "3 negation",
]

def main():
    data = load_eval_from_csv("evaluation_data.csv")

    k = 10

    models_to_test = {
        "0 BM25": (None, "bm25"),
        "1 Colbert (colbertv2.0)": ("colbert-ir/colbertv2.0", "colbert"),
        "2 Colbert Domain Adapted (bert->sft->colbert)": ("colbert_transplant_two_stage_runs/dataset_then_general/stage2", "transplant"),

        "3 CrossEncoder (mxbai-rerank-lg-v1)": ("mixedbread-ai/mxbai-rerank-large-v1", "cross_encoder"),
        "4 Dense (bge-lg)": ("BAAI/bge-large-en-v1.5", "dense"),

        "5 Dense Domain Adapted (bge-lg->sft)": ("bge_results/bge_large_stage2", "dense"),
    }

    all_stats = []

    scorers = build_scorers(models_to_test)

    try:
        for item in data["queries"]:
            q_text = item["text"]
            gold = item["gold_standard"]
            docs = list(gold.keys())
            docs = shuffle_docs(docs, seed=42)

            print(f"\n{'='*80}\nQUERY: {q_text}\n{'='*80}")

            for name, scorer in scorers.items():
                try:
                    scores_dict = scorer.score(q_text, docs)
                    metrics, ranked_list = get_metrics(gold, scores_dict, k=k)

                    query_category = item.get("category", "unknown")

                    metrics["Model"] = name
                    metrics["QueryID"] = item["query_id"]
                    metrics["QueryText"] = q_text
                    metrics["Category"] = query_category
                    all_stats.append(metrics)

                    print(f"\n[{name}] TOP {k} RETRIEVED:")
                    print(f"{'Rank':<5} | {'Score':<6} | {'Gold':<4} | {'Document Content'}")
                    print("-" * 90)
                    for i, doc in enumerate(ranked_list[:k]):
                        g_score = gold.get(doc, 0)
                        m_score = scores_dict[doc]
                        display_doc = (doc[:60] + "..") if len(doc) > 60 else doc
                        print(f"{i+1:<5} | {m_score:>6.2f} | {g_score:^4} | {display_doc}")

                    print(
                        f"Metrics @{k}: "
                        f"P: {metrics[f'P@{k}']:.2f}, "
                        f"R: {metrics[f'R@{k}']:.2f}, "
                        f"F1: {metrics[f'F1@{k}']:.2f}, "
                        f"nDCG: {metrics[f'nDCG@{k}']:.3f}, "
                        #f"nDCG@3: {metrics[f'nDCG@3']:.3f}, "
                        # f"R@R: {metrics['R@R']:.3f}, "
                        # f"R={metrics['NumRelevant']}\n"
                    )

                except Exception as e:
                    print(f"⚠️ Failed to evaluate model [{name}]: {e}")

    finally:
        for scorer in scorers.values():
            scorer.cleanup()

        torch.cuda.empty_cache()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print("\n\n" + "=" * 30 + " GLOBAL SUMMARY " + "=" * 30)
    summary_df = pd.DataFrame(all_stats)

    if not summary_df.empty:
        metric_cols = [
            f"P@{k}",
            f"R@{k}",
            f"F1@{k}",
            # "R@R",
            f"nDCG@{k}",
            #"nDCG@3",
        ]

        summary_df["Category"] = pd.Categorical(
            summary_df["Category"],
            categories=CATEGORY_ORDER,
            ordered=True
        )

        overall_summary = (
            summary_df.groupby("Model")[metric_cols]
            .mean()
            .round(3)
            .sort_values(by=f"nDCG@{k}", ascending=False)
        )
        print(overall_summary)

        print("\n\n" + "=" * 30 + " SUMMARY BY CATEGORY " + "=" * 30)
        category_summary = (
            summary_df.groupby(["Category", "Model"], observed=True)[metric_cols]
            .mean()
            .round(3)
            .sort_index(level=["Category", "Model"])
        )
        print(category_summary)

        print("\n\n" + "=" * 30 + " QUERY COUNTS BY CATEGORY " + "=" * 30)
        query_counts = (
            summary_df[["QueryID", "QueryText", "Category"]]
            .drop_duplicates()
            .groupby("Category", observed=False)
            .size()
            .reindex(CATEGORY_ORDER, fill_value=0)
        )
        print(query_counts)

        # make an output dir once
        os.makedirs(RESULTS_OUTPUT, exist_ok=True)

        # 1) Full row-level results: one row per query x model
        summary_df.to_csv(RESULTS_OUTPUT + "/per_query_model_results.csv", index=False)

        # 2) Overall summary: one row per model
        overall_summary.reset_index().to_csv(RESULTS_OUTPUT + "/overall_summary.csv", index=False)

        # 3) Category summary: one row per category x model
        category_summary.reset_index().to_csv(RESULTS_OUTPUT +  "/category_summary.csv", index=False)

        # 4) Query counts by category
        query_counts.rename("query_count").reset_index().to_csv(
            RESULTS_OUTPUT + "/query_counts_by_category.csv",
            index=False
        )

        print("\nSaved CSVs to " + RESULTS_OUTPUT)


if __name__ == "__main__":
    main()