#!/usr/bin/env python3
import os
import random
from collections import defaultdict, deque
from typing import Iterator, List

import pandas as pd
from datasets import Dataset
from torch.utils.data import BatchSampler, DataLoader
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

# =========================
# CONFIG
# =========================

BASE_MODEL = "BAAI/bge-large-en-v1.5"

GENERAL_PAIRS_CSV = "data/general_pairs.csv"
DATASET_PAIRS_CSV = "data/dataset_pairs.csv"

OUTPUT_ROOT = "bge_results"
STAGE1_OUTPUT = os.path.join(OUTPUT_ROOT, "bge_large_stage1")
STAGE2_OUTPUT = os.path.join(OUTPUT_ROOT, "bge_large_stage2")

SEED = 42
MAX_QUERY_CHARS = 256
MAX_DOC_CHARS = 1400
VAL_SPLIT = 0.1

QUERY_PREFIX = ""
DOC_PREFIX = ""

STAGE1_EPOCHS = 2
STAGE1_BATCH_SIZE = 32
STAGE1_LR = 1e-5

STAGE2_EPOCHS = 2
STAGE2_BATCH_SIZE = 16
STAGE2_LR = 1e-5


# =========================
# UTILS
# =========================

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_text(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ").strip()
    text = " ".join(text.split())
    return text[:max_chars]


def add_prefix_query(text: str) -> str:
    return f"{QUERY_PREFIX}{text}" if QUERY_PREFIX else text


def add_prefix_doc(text: str) -> str:
    return f"{DOC_PREFIX}{text}" if DOC_PREFIX else text


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_dataset(dataset, val_split=0.2, seed=42, name="dataset"):
    from sklearn.model_selection import train_test_split

    unique_docs = list(dict.fromkeys(dataset["document"]))

    train_docs, val_docs = train_test_split(
        unique_docs,
        test_size=val_split,
        random_state=seed,
    )

    train_docs = set(train_docs)

    train_dataset = dataset.filter(
        lambda x: x["document"] in train_docs
    )
    val_dataset = dataset.filter(
        lambda x: x["document"] not in train_docs
    )

    assert set(train_dataset["document"]).isdisjoint(
        set(val_dataset["document"])
    )

    print(f"{name}: {len(train_dataset)} train, "
          f"{len(val_dataset)} validation")

    return train_dataset, val_dataset


# =========================
# CUSTOM BATCH SAMPLER
# =========================

class UniqueDocumentBatchSampler(BatchSampler):
    """
    Build batches with at most one example per unique document.

    This prevents MultipleNegativesRankingLoss from seeing the same document
    twice in one batch when multiple queries map to the same document.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
        document_column: str = "document",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.document_column = document_column
        self.epoch = 0

        if document_column not in dataset.column_names:
            raise ValueError(
                f"Dataset is missing required column '{document_column}'. "
                f"Available columns: {dataset.column_names}"
            )

        grouped = defaultdict(list)
        docs = dataset[document_column]
        for idx, doc in enumerate(docs):
            grouped[doc].append(idx)

        self.doc_to_indices = dict(grouped)
        self.unique_docs = list(self.doc_to_indices.keys())
        self.num_examples = len(dataset)

        print(
            f"UniqueDocumentBatchSampler: {len(self.unique_docs):,} unique documents "
            f"across {self.num_examples:,} examples"
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        # Shuffle examples within each document bucket
        per_doc_queues = {}
        for doc, indices in self.doc_to_indices.items():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            per_doc_queues[doc] = deque(shuffled)

        # Active documents that still have remaining examples
        active_docs = self.unique_docs[:]
        emitted = 0

        while active_docs:
            rng.shuffle(active_docs)

            # Take up to batch_size unique docs for this batch
            batch_docs = active_docs[:self.batch_size]
            remaining_docs = active_docs[self.batch_size:]

            batch = []
            recycled_docs = []

            for doc in batch_docs:
                q = per_doc_queues[doc]
                if not q:
                    continue

                batch.append(q.popleft())
                emitted += 1

                # Put doc back into circulation if it still has more examples
                if q:
                    recycled_docs.append(doc)

            # Keep unvisited docs, then re-add docs that still have leftover examples
            active_docs = remaining_docs + recycled_docs

            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                yield batch

        if not self.drop_last and emitted != self.num_examples:
            raise RuntimeError(
                f"Sampler emitted {emitted} examples, expected {self.num_examples}"
            )

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_examples // self.batch_size
        return (self.num_examples + self.batch_size - 1) // self.batch_size

class DocSafeSentenceTransformerTrainer(SentenceTransformerTrainer):
    """
    SentenceTransformerTrainer that uses a document-unique batch sampler.
    """

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        sampler = UniqueDocumentBatchSampler(
            dataset=train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
            document_column="document",
        )

        return DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        sampler = UniqueDocumentBatchSampler(
            dataset=eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            drop_last=False,
            seed=self.args.seed,
            document_column="document",
        )

        return DataLoader(
            eval_dataset,
            batch_sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# =========================
# LOAD DATA
# =========================

def load_general_pairs(path: str) -> Dataset:
    df = pd.read_csv(path)

    required = {"query", "document"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in general pairs CSV: {missing}")

    df = df.dropna(subset=["query", "document"]).copy()

    df["query"] = df["query"].map(lambda x: add_prefix_query(clean_text(x, MAX_QUERY_CHARS)))
    df["document"] = df["document"].map(lambda x: add_prefix_doc(clean_text(x, MAX_DOC_CHARS)))

    df = df[(df["query"].str.len() > 0) & (df["document"].str.len() > 0)]
    df = df.drop_duplicates(subset=["query", "document"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} general pairs from {path}")
    return Dataset.from_pandas(df[["query", "document"]], preserve_index=False)


def load_dataset_pairs(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)

    required = {"query", "document"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset pairs CSV: {missing}")

    df = df.dropna(subset=["query", "document"]).copy()

    df["query"] = df["query"].map(lambda x: add_prefix_query(clean_text(x, MAX_QUERY_CHARS)))
    df["document"] = df["document"].map(lambda x: add_prefix_doc(clean_text(x, MAX_DOC_CHARS)))

    df = df[
        (df["query"].str.len() > 0)
        & (df["document"].str.len() > 0)
    ]

    df = df.drop_duplicates(subset=["query", "document"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} dataset pairs from {csv_path}")
    return Dataset.from_pandas(df[["query", "document"]], preserve_index=False)


# =========================
# STAGE 1: GENERAL PAIRS
# =========================

def train_stage1_general_pairs(
    input_model: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
) -> str:
    ensure_dir(output_dir)

    model = SentenceTransformer(input_model)
    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=STAGE1_EPOCHS,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        per_device_eval_batch_size=STAGE1_BATCH_SIZE,
        learning_rate=STAGE1_LR,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = DocSafeSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    print("\n=== Stage 1: training on general pairs ===")
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Saved stage 1 model to: {output_dir}")
    return output_dir


# =========================
# STAGE 2: DATASET PAIRS
# =========================

def train_stage2_dataset_pairs(
    input_model: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
) -> str:
    ensure_dir(output_dir)

    model = SentenceTransformer(input_model)
    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=STAGE2_EPOCHS,
        per_device_train_batch_size=STAGE2_BATCH_SIZE,
        per_device_eval_batch_size=STAGE2_BATCH_SIZE,
        learning_rate=STAGE2_LR,
        warmup_ratio=0.03,
        fp16=True,
        bf16=False,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = DocSafeSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    print("\n=== Stage 2: training on dataset pairs ===")
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Saved stage 2 model to: {output_dir}")
    return output_dir


# =========================
# MAIN
# =========================

def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_ROOT)

    if not os.path.exists(GENERAL_PAIRS_CSV):
        raise FileNotFoundError(f"Missing file: {GENERAL_PAIRS_CSV}")
    if not os.path.exists(DATASET_PAIRS_CSV):
        raise FileNotFoundError(f"Missing file: {DATASET_PAIRS_CSV}")

    general_pairs = load_general_pairs(GENERAL_PAIRS_CSV)
    dataset_pairs = load_dataset_pairs(DATASET_PAIRS_CSV)

    dataset_train, dataset_val = split_dataset(
        dataset_pairs,
        val_split=VAL_SPLIT,
        seed=SEED,
        name="Dataset pairs",
    )

    general_train, general_val = split_dataset(
        general_pairs,
        val_split=VAL_SPLIT,
        seed=SEED,
        name="General pairs",
    )

    stage1_model_path = train_stage1_general_pairs(
        input_model=BASE_MODEL,
        train_dataset=general_train,
        eval_dataset=general_val,
        output_dir=STAGE1_OUTPUT,
    )

    final_model_path = train_stage2_dataset_pairs(
        input_model=stage1_model_path,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        output_dir=STAGE2_OUTPUT,
    )

    print("\nDone.")
    print(f"Final model path: {final_model_path}")


if __name__ == "__main__":
    main()