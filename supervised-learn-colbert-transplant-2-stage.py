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
    models,
)

os.environ["WANDB_MODE"] = "disabled"

# =========================
# CONFIG
# =========================

BASE_MODEL_PATH = "bert-base-uncased"
#BASE_MODEL_PATH = "archaeobert_mlm/adapter"

GENERAL_PAIRS_CSV = "data/general_pairs.csv"
DATASET_PAIRS_CSV = "data/dataset_pairs.csv"

OUTPUT_ROOT = "colbert_transplant_two_stage_runs"

MAX_SEQ_LENGTH = 256
USE_NORMALIZE = False

SEED = 42
VAL_SPLIT = 0.1

EPOCHS_PER_STAGE = 2
WARMUP_RATIO = 0.1
BATCH_SIZE = 16


# =========================
# HELPERS
# =========================
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(model_path: str, max_seq_length: int = 256, use_normalize: bool = True):
    """
    Load either:
    - a full SentenceTransformer model directory
    - or a Hugging Face transformer checkpoint
    """
    if os.path.exists(os.path.join(model_path, "modules.json")):
        model = SentenceTransformer(model_path)
    else:
        word_embedding_model = models.Transformer(
            model_path,
            max_seq_length=max_seq_length,
        )

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        modules_list = [word_embedding_model, pooling_model]
        if use_normalize:
            modules_list.append(models.Normalize())

        model = SentenceTransformer(modules=modules_list)

    return model


def load_examples_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find training file: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["query", "document"]).copy()

    for col in ["query", "document"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["query"] != "") &
        (df["document"] != "")
    ].copy()

    df = df.drop_duplicates(subset=["query", "document"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows from {csv_path}")
    return df


def split_df(df, val_split=0.2, seed=42, name="dataset"):
    from sklearn.model_selection import train_test_split

    unique_docs = df["document"].drop_duplicates()

    train_docs, val_docs = train_test_split(
        unique_docs,
        test_size=val_split,
        random_state=seed,
    )

    train_df = df[df["document"].isin(train_docs)].copy()
    val_df = df[df["document"].isin(val_docs)].copy()

    assert set(train_df["document"]).isdisjoint(
        set(val_df["document"])
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def df_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["query", "document"]], preserve_index=False)


def load_and_split_dataset(csv_path: str, val_split: float = 0.2, seed: int = 42, name: str = "dataset"):
    df = load_examples_df(csv_path)
    train_df, val_df = split_df(df, val_split=val_split, seed=seed, name=name)
    return df_to_dataset(train_df), df_to_dataset(val_df)


# =========================
# CUSTOM BATCH SAMPLER
# =========================
class UniqueDocumentBatchSampler(BatchSampler):
    """
    Yield batches with at most one example per unique document.

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

        per_doc_queues = {}
        for doc, indices in self.doc_to_indices.items():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            per_doc_queues[doc] = deque(shuffled)

        active_docs = self.unique_docs[:]
        emitted = 0

        while active_docs:
            rng.shuffle(active_docs)

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

                if q:
                    recycled_docs.append(doc)

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
    SentenceTransformerTrainer that uses a document-unique batch sampler
    for train/eval datasets.
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
# TRAINING
# =========================
def train_one_stage(
    base_model_path: str,
    train_csv: str,
    output_path: str,
    epochs: int = 4,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    max_seq_length: int = 256,
    use_normalize: bool = True,
    val_split: float = 0.2,
    seed: int = 42,
):
    print("\n" + "=" * 100)
    print("TRAINING STAGE")
    print(f"Base model:   {base_model_path}")
    print(f"Train file:   {train_csv}")
    print(f"Output path:  {output_path}")
    print("=" * 100)

    model = build_model(
        model_path=base_model_path,
        max_seq_length=max_seq_length,
        use_normalize=use_normalize,
    )

    train_dataset, val_dataset = load_and_split_dataset(
        train_csv,
        val_split=val_split,
        seed=seed,
        name=os.path.basename(train_csv),
    )

    loss = losses.MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-5,
        warmup_ratio=warmup_ratio,
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
        seed=seed,
        remove_unused_columns=False,
    )

    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples:   {len(val_dataset):,}")
    print(f"Batch size:     {batch_size}")
    print(f"Epochs:         {epochs}")
    print(f"Warmup ratio:   {warmup_ratio}")

    trainer = DocSafeSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    trainer.train()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Saved stage model to: {output_path}")

    del model


def run_two_stage_experiment(
    run_name: str,
    stage1_csv: str,
    stage2_csv: str,
    base_model_path: str,
):
    print("\n" + "#" * 120)
    print(f"RUNNING EXPERIMENT: {run_name}")
    print("#" * 120)

    stage1_output = os.path.join(OUTPUT_ROOT, run_name, "stage1")
    stage2_output = os.path.join(OUTPUT_ROOT, run_name, "stage2")

    # Stage 1
    train_one_stage(
        base_model_path=base_model_path,
        train_csv=stage1_csv,
        output_path=stage1_output,
        epochs=2,
        batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        max_seq_length=MAX_SEQ_LENGTH,
        use_normalize=USE_NORMALIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    # Stage 2 starts from Stage 1 model
    train_one_stage(
        base_model_path=stage1_output,
        train_csv=stage2_csv,
        output_path=stage2_output,
        epochs=4,
        batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        max_seq_length=MAX_SEQ_LENGTH,
        use_normalize=USE_NORMALIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    print(f"\nFinished experiment: {run_name}")
    print(f"Final model: {stage2_output}")
    print(f"If transplanting into ColBERT, also check: {os.path.join(stage2_output, '0_Transformer')}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    # general -> dataset
    # run_two_stage_experiment(
    #     run_name="general_then_dataset",
    #     stage1_csv=GENERAL_PAIRS_CSV,
    #     stage2_csv=DATASET_PAIRS_CSV,
    #     base_model_path=BASE_MODEL_PATH,
    # )

    # dataset -> general
    run_two_stage_experiment(
        run_name="dataset_then_general",
        stage1_csv=DATASET_PAIRS_CSV,
        stage2_csv=GENERAL_PAIRS_CSV,
        base_model_path=BASE_MODEL_PATH,
    )