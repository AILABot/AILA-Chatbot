"""
Embedding model fine-tuning with Matryoshka evaluation.

This module defines utilities to fine-tune Sentence-Transformers embedding models
across Matryoshka dimensions, evaluate them with IR metrics, and push the best
checkpoint to the Hugging Face Hub.

Sphinx:
    - Docstrings follow **Google style** for seamless `sphinx.ext.napoleon` parsing.
    - Public functions and methods document parameters, returns, and behavior.

Environment:
    - Requires valid credentials in `backend.database.config.config.settings`
      (uses `settings.HF_TOKEN` for Hugging Face).
"""

import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
import pandas as pd
from typing import List, Dict, Tuple
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
import datasets
from backend.database.config.config import settings
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback
from pathlib import Path
from huggingface_hub import create_repo, upload_folder
import gc
import os


def cuda_clean() -> None:
    """Free Python and CUDA memory to stabilize long training runs.

    This helper:
      - Forces Python GC.
      - Empties CUDA caches (if available).
      - Calls `torch.cuda.ipc_collect()` to release inter-process memory.

    Returns:
        None
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class EmbeddingFinetuning:
    """Fine-tune embedding models with Matryoshka losses and IR evaluation.

    The workflow:
      1) Load/augment a base QA dataset.
      2) Build IR evaluator(s) across Matryoshka dimensions.
      3) Train each model with `MatryoshkaLoss(MultipleNegativesRankingLoss)`.
      4) Evaluate before/after.
      5) Push the best checkpoint to the Hugging Face Hub.

    Attributes:
        device: `"cuda"` if available, else `"cpu"`.
        models: List of model IDs to consider (defaults provided).
        model_dimensions: Mapping from model ID to Matryoshka dims (in descending order).
    """

    def __init__(self) -> None:
        """Initialize defaults and device selection."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models = [
            'IoannisKat1/multilingual-e5-large-legal-matryoshka',
            "IoannisKat1/legal-bert-base-uncased-legal-matryoshka",
            'IoannisKat1/modernbert-embed-base-legal-matryoshka-2',
            'IoannisKat1/bge-m3-legal-matryoshka'
        ]

        self.model_dimensions = {
            'IoannisKat1/multilingual-e5-large-legal-matryoshka': [1024, 768, 512, 256, 128, 64],
            "IoannisKat1/legal-bert-base-uncased-legal-matryoshka": [768, 512, 256, 128, 64],
            'IoannisKat1/modernbert-embed-base-legal-matryoshka-2': [768, 512, 256, 128, 64],
            'IoannisKat1/bge-m3-legal-matryoshka': [768, 512, 256, 128, 64],
        }

        # Candidate models to fine-tune (Matryoshka-capable heads)
        # self.models = [
        #     "IoannisKat1/multilingual-e5-large-new2",
        #     "IoannisKat1/legal-bert-base-uncased-new2",
        #     "IoannisKat1/modernbert-embed-base-new2",
        #     "IoannisKat1/bge-m3-new2",
        # ]

        # # Matryoshka dimensions per model
        # self.model_dimensions: Dict[str, List[int]] = {
        #     "IoannisKat1/multilingual-e5-large-new2": [1024, 768, 512, 256, 128, 64],
        #     "IoannisKat1/legal-bert-base-uncased-new2": [768, 512, 256, 128, 64],
        #     "IoannisKat1/modernbert-embed-base-new2": [768, 512, 256, 128, 64],
        #     "IoannisKat1/bge-m3-new2": [768, 512, 256, 128, 64],
        # }

    def data_preparation(self, new_data: List | None) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, List[int]], datasets.Dataset]:
        """Load the base CSV, optionally merge new samples, and build IR structures.

        Input CSV (Windows path style in original setup):
            `backend\\reinforcement_learning\\new_queries.csv`

        Expected columns in the combined dataframe:
            - `anchor` (str): query text
            - `positive` (str): relevant passage
            - `global_chunk_id` (int): group identifier for relevance
            - `id` (int): unique row id

        If `new_data` is provided, we append rows with:
            - `anchor`: `data['example']['query']`
            - `positive`: first of `data['example']['answer']`
            - `global_chunk_id`: best guess from base df (left as-is per original logic)
            - `id`: incremental index appended to the end

        Args:
            new_data: Optional list of additional examples to merge into the base set.

        Returns:
            A tuple:
                - `queries` (dict[int, str]): query_id -> query_text (from test split)
                - `corpus` (dict[int, str]): doc_id -> passage_text (train + test)
                - `relevant_docs` (dict[int, list[int]]): query_id -> list of relevant doc_ids
                - `train_dataset` (datasets.Dataset): training subset with columns
                  at least `anchor`, `positive`, `global_chunk_id`, `id`
        """
        df_base = pd.read_csv('backend\\reinforcement_learning\\new_queries.csv')

        if new_data:
            count = len(df_base)
            new_data_list = []
            for data in new_data:
                # NOTE: Retain original logic: choose an existing global_chunk_id if present;
                # otherwise fall back to the current max.
                global_chunk_id = max(df_base['global_chunk_id'])
                for i in range(len(df_base)):
                    if df_base['global_chunk_id'][i] == data['example']['answer']:
                        global_chunk_id = df_base['global_chunk_id'][i]
                new_data_list.append([data['example']['query'], data['example']['answer'][0], global_chunk_id, count])
                count += 1

            df_new = pd.DataFrame(new_data_list, columns=['anchor', 'positive', 'global_chunk_id', 'id'])
            df_combined = pd.concat([df_base.iloc[:, 1:], df_new], ignore_index=True)
        else:
            df_combined = df_base

        print(df_combined)

        dataset = datasets.Dataset.from_pandas(df_combined)
        dataset = dataset.shuffle(seed=43).train_test_split(test_size=0.2)

        dataset['train'].to_json('train_dataset.json', orient='records')
        dataset['test'].to_json('test_dataset.json', orient='records')
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        # Build IR structures
        corpus_dataset = datasets.concatenate_datasets([train_dataset, test_dataset])
        corpus = dict(zip(corpus_dataset['id'], corpus_dataset['positive']))
        queries = dict(zip(test_dataset['id'], test_dataset['anchor']))

        relevant_docs: Dict[int, List[int]] = {}
        for q_id, global_chunk_id in zip(test_dataset['id'], test_dataset['global_chunk_id']):
            if q_id not in relevant_docs:
                relevant_docs[q_id] = []
            matching_corpus_ids = [
                cid for cid, chunk in zip(corpus_dataset['id'], corpus_dataset['global_chunk_id'])
                if chunk == global_chunk_id
            ]
            relevant_docs[q_id].extend(matching_corpus_ids)

        return queries, corpus, relevant_docs, train_dataset

    def _build_evaluator(
        self,
        queries: Dict[int, str],
        corpus: Dict[int, str],
        relevant_docs: Dict[int, List[int]],
        matryoshka_dimensions: List[int],
        eval_batch_size: int = 16,
    ) -> SequentialEvaluator:
        """Create a `SequentialEvaluator` over multiple Matryoshka dimensions.

        For each dimension, we construct an `InformationRetrievalEvaluator`
        that computes cosine-based IR metrics (ndcg, mrr, map, accuracy@k, etc.).

        Args:
            queries: Mapping of query ids to texts.
            corpus: Mapping of document ids to texts.
            relevant_docs: Mapping of query ids to lists of relevant document ids.
            matryoshka_dimensions: Target embedding dimensions to truncate to.
            eval_batch_size: Batch size for evaluation forward passes.

        Returns:
            A `SequentialEvaluator` that runs each per-dimension IR evaluation.
        """
        mats = []
        for dim in matryoshka_dimensions:
            mats.append(
                InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=f'dim_{dim}',
                    truncate_dim=dim,
                    score_functions={"cosine": cos_sim},
                    batch_size=eval_batch_size,
                    show_progress_bar=False,
                )
            )
        return SequentialEvaluator(mats)

    def model_finetuning(self, data: list, models_names: List[str]) -> None:
        """Fine-tune one or more embedding models and report IR metrics.

        Steps:
            1) Prepare data & IR evaluators.
            2) For each `model_name`:
                - Load model with memory-friendly kwargs.
                - Optional gradient checkpointing (if supported).
                - Evaluate base performance.
                - Train with `MatryoshkaLoss(MultipleNegativesRankingLoss)`.
                - Evaluate finetuned performance.
                - Push best checkpoint to the Hub.

        Args:
            data: Optional augmentation samples for `data_preparation`.
            models_names: Subset of `self.models` to train.

        Returns:
            None
        """
        queries, corpus, relevant_docs, train_dataset = self.data_preparation(data)
        print(models_names)

        for model_name in models_names:
            cuda_clean()

            matryoshka_dimensions = self.model_dimensions[model_name]
            evaluator = self._build_evaluator(queries, corpus, relevant_docs, matryoshka_dimensions, eval_batch_size=16)

            # Choose dtype based on device capability
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

            # Workaround for torch.compile error during loading
            os.environ['TORCH_COMPILE_ENABLE'] = '0'
            model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs={
                    "torch_dtype": torch_dtype,
                    "attn_implementation": "sdpa",
                },
                model_card_data=SentenceTransformerModelCardData(
                    language="en",
                    license="apache-2.0",
                    model_name=f"{model_name.split('/')[1]}",
                ),
            )
            # Unset after load
            del os.environ['TORCH_COMPILE_ENABLE']

            # Optional: gradient checkpointing
            try:
                if hasattr(model, "auto_model") and hasattr(model.auto_model, "gradient_checkpointing_enable"):
                    model.auto_model.gradient_checkpointing_enable()
            except Exception:
                pass

            # Baseline evaluation
            base_results = evaluator(model)
            self._print_results(f"Base Model Evaluation Results {model_name}", matryoshka_dimensions, base_results)

            # Loss: Matryoshka wrapper over MNRS
            base_loss = MultipleNegativesRankingLoss(model)
            train_loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=matryoshka_dimensions)

            # Train
            self.model_training(
                model=model,
                model_name=model_name,
                train_dataset=train_dataset,
                train_loss=train_loss,
                evaluator=evaluator,
                matryoshka_dimensions=matryoshka_dimensions,
                train_epochs=20,
                train_batch_size=8,       # micro-batch
                gradient_step=4,          # grad accumulation
                eval_batch_size=8,
                learning_rate=3e-5,
            )

            # Cleanup between models
            del model
            cuda_clean()

    def _print_results(self, title: str, matryoshka_dimensions: List[int], results: Dict[str, float]) -> None:
        """Pretty-print IR metrics across Matryoshka dimensions.

        Args:
            title: Section title to print.
            matryoshka_dimensions: Ordered list of dims used during eval.
            results: Dictionary returned by `InformationRetrievalEvaluator`/`SequentialEvaluator`.

        Returns:
            None
        """
        print(f"\n{title}")
        print("-" * 85)
        if matryoshka_dimensions[0] == 1024:
            head = f"{'Metric':15} {'1024d':>12} {'768d':>12} {'512d':>12} {'256d':>12} {'128d':>12} {'64d':>12}"
        else:
            head = f"{'Metric':15} {'768d':>12} {'512d':>12} {'256d':>12} {'128d':>12} {'64d':>12}"
        print(head)
        print("-" * 85)

        metrics = [
            'ndcg@10','mrr@10','map@100','accuracy@1','accuracy@3','accuracy@5','accuracy@10',
            'precision@1','precision@3','precision@5','precision@10','recall@1','recall@3','recall@5','recall@10'
        ]
        for metric in metrics:
            vals = []
            for dim in matryoshka_dimensions:
                key = f"dim_{dim}_cosine_{metric}"
                vals.append(results[key])
            name = f"=={metric}==" if metric == "ndcg@10" else metric
            print(f"{name:15}", *[f"{v:12.4f}" for v in vals])
        print("-" * 85)
        print(f"seq_score: {results['sequential_score']:1f}")

    def push_best_to_hub(self, model: SentenceTransformer, trainer: SentenceTransformerTrainer, model_name: str, namespace: str = "IoannisKat1") -> None:
        """Save and upload the best checkpoint to the Hugging Face Hub.

        The method:
            1) Locates the best checkpoint from the trainer (or saves the in-memory model).
            2) Creates (or reuses) the target Hub repo.
            3) Uploads the saved folder.

        Args:
            model: The trained `SentenceTransformer` model.
            trainer: The `SentenceTransformerTrainer` containing training state.
            model_name: Original model name/id (used for local folder naming).
            namespace: Hub namespace/org (default: `"IoannisKat1"`).

        Returns:
            None
        """
        best_ckpt = trainer.state.best_model_checkpoint  # may be None if no eval improvement
        out_dir = model_name.replace("/", "-")
        if best_ckpt is None:
            save_dir = Path(out_dir) / "final"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))
        else:
            save_dir = Path(best_ckpt) / "export"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))

        repo_basename = model_name.split("/")[1]
        repo_id = f"{namespace}/{repo_basename}-new2"

        token = settings.HF_TOKEN

        create_repo(repo_id, private=False, exist_ok=True, token=token)
        upload_folder(
            repo_id=repo_id,
            folder_path=str(save_dir),
            commit_message="Add finetuned model",
            token=token,
        )
        print(f"Pushed {save_dir} -> {repo_id}")

    def model_training(
        self,
        model: SentenceTransformer,
        model_name: str,
        train_dataset: datasets.Dataset,
        train_loss,
        evaluator: SequentialEvaluator,
        matryoshka_dimensions: List[int],
        train_epochs: int = 20,
        train_batch_size: int = 8,
        gradient_step: int = 4,
        eval_batch_size: int = 8,
        learning_rate: float = 3e-5,
    ) -> None:
        """Train a model with Matryoshka loss and evaluate each epoch.

        Args:
            model: The `SentenceTransformer` instance to fine-tune.
            model_name: Model ID (used for output directory naming).
            train_dataset: Training dataset (expects columns `anchor`, `positive`).
            train_loss: Loss function, typically `MatryoshkaLoss`.
            evaluator: `SequentialEvaluator` built by `_build_evaluator`.
            matryoshka_dimensions: Dimensions used in Matryoshka loss/eval.
            train_epochs: Number of epochs (default: 20).
            train_batch_size: Micro-batch size (default: 8).
            gradient_step: Gradient accumulation steps (default: 4).
            eval_batch_size: Evaluation batch size (default: 8).
            learning_rate: Learning rate for AdamW (default: 3e-5).

        Returns:
            None
        """
        args = SentenceTransformerTrainingArguments(
            output_dir=model_name.replace('/', '-'),
            num_train_epochs=train_epochs,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_step,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=0.1,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            bf16=True,
            fp16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_dim_128_cosine_ndcg@10",
            report_to="none",
            dataloader_pin_memory=True,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset.select_columns(["anchor", "positive"]),
            loss=train_loss,
            evaluator=evaluator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        ft_results = evaluator(model)
        self._print_results(f"Finetuned Embedding {model_name} Evaluation Results", matryoshka_dimensions, ft_results)

        self.push_best_to_hub(model, trainer, model_name)
