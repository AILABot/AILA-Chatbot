"""
Cross-Encoder fine-tuning utilities (Sentence-Transformers).

This module exposes :class:`CrossEncoderFinetuning` for preparing data, mining
hard negatives, training a CrossEncoder with strong evaluators, and pushing the
best checkpoint to the Hugging Face Hub.

Sphinx:
    - Docstrings follow **Google style** for `sphinx.ext.napoleon`.

Environment:
    - Requires a valid Hugging Face token at
      ``backend.database.config.config.settings.HF_TOKEN``.
"""

from typing import List, Optional, Dict, Any
import os
from torch.utils.data import DataLoader  # noqa: F401 (kept for downstream extension)
from sentence_transformers import InputExample, losses, evaluation, models  # noqa: F401
from huggingface_hub import create_repo, upload_folder
from sentence_transformers import CrossEncoder, CrossEncoderTrainer, SentenceTransformer
from datasets import Dataset
from pathlib import Path
import pandas as pd
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
from backend.database.config.config import settings
# from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator  # Optional

class CrossEncoderFinetuning:
    """Fine-tune a `sentence_transformers.CrossEncoder` with hard-negative mining.

    Workflow:
        1) Load a base dataset of (anchor, positive) pairs.
        2) Optionally add user-supplied positives/negatives.
        3) Mine hard negatives using a Bi-Encoder.
        4) Train a CrossEncoder and evaluate with classification + reranking evaluators.
        5) Push the best checkpoint to the Hugging Face Hub.

    Args:
        model_id: Model repository or path used to initialize the CrossEncoder.
            Defaults to ``'IoannisKat1/bge-reranker-ft-new2'``.

    Attributes:
        model_id: The base model identifier.
        model: The instantiated `CrossEncoder` model.
    """

    def __init__(self, model_id: str = 'IoannisKat1/bge-reranker-ft-new2') -> None:
        self.model_id = model_id
        self.model = CrossEncoder(self.model_id, num_labels=1, max_length=512)

    def data_preparation(self, new_data: Optional[List[Dict[str, Any]]] = None) -> Dataset:
        """Assemble a labeled dataset for CrossEncoder training.

        Loads a base CSV at ``backend\\reinforcement_learning\\new_queries.csv``
        with columns:
            - ``anchor``: query text (str)
            - ``positive``: positive (relevant) response text (str)

        Optionally augments with ``new_data`` entries shaped as:
        ```
        [
          {
            "example": {
              "query": "<str>",
              "answer": ["<positive str>", ...],
              "negative_answer": ["<negative str>", ...]
            }
          },
          ...
        ]
        ```

        Args:
            new_data: Optional additional samples with positives and negatives.

        Returns:
            Dataset: HuggingFace `Dataset` with columns
            ``{"query": str, "response": str, "label": int}``.
        """
        df_base = pd.read_csv('backend\\reinforcement_learning\\new_queries.csv')

        queries: List[str] = []
        labels: List[int] = []
        answers: List[str] = []

        # Base positives (anchor -> positive, label=1)
        for i in range(len(df_base)):
            query = df_base['anchor'][i]
            answers_list = [df_base['positive'][i]]
            labels_list = [1]
            queries += [query for _ in range(len(labels_list))]
            answers += answers_list
            labels += labels_list

        # Optional: add new positives + negatives
        if new_data:
            for i in range(len(new_data)):
                ex = new_data[i]['example']
                query = ex['query']
                answers_list = list(ex.get('answer', [])) + list(ex.get('negative_answer', []))
                labels_list = [1 for _ in ex.get('answer', [])] + [0 for _ in ex.get('negative_answer', [])]
                queries += [query for _ in range(len(labels_list))]
                answers += answers_list
                labels += labels_list

        dataset = Dataset.from_dict({'query': queries, 'response': answers, 'label': labels})
        return dataset

    def push_best_to_hub(self, model: CrossEncoder, trainer: CrossEncoderTrainer, model_name: str, namespace: str = "IoannisKat1") -> None:
        """Save and upload the best checkpoint to the Hugging Face Hub.

        The method:
            1) Locates the best checkpoint from the trainer (or saves the in-memory model).
            2) Creates (or reuses) the target Hub repo.
            3) Uploads the saved folder.

        Args:
            model: The trained `CrossEncoder` model.
            trainer: The `CrossEncoderTrainer` (for best checkpoint location).
            model_name: Local output folder name / base for repo basename.
            namespace: Hugging Face namespace (organization or user).

        Returns:
            None
        """
        best_ckpt = trainer.state.best_model_checkpoint  # may be None if no eval improvement
        out_dir = model_name
        if best_ckpt is None:
            save_dir = Path(out_dir) / "final"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))
        else:
            save_dir = Path(best_ckpt) / "export"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))

        repo_basename = model_name
        repo_id = f"{namespace}/{repo_basename}-new2"

        token = settings.HF_TOKEN

        create_repo(repo_id, private=False, exist_ok=True, token=token)
        upload_folder(
            repo_id=repo_id,
            folder_path=str(save_dir),
            commit_message="Add finetuned model",
            token=token
        )
        print(f"Pushed {save_dir} -> {repo_id}")

    def cross_encoder_tuning(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        eval_split: float = 0.2,
        output_path: str = "bge-reranker-ft",
    ) -> None:
        """Fine-tune the CrossEncoder with hard-negative mining and strong evaluators.

        Pipeline:
            1) Build labeled dataset via :meth:`data_preparation`.
            2) Split train/test; mine hard negatives with a Bi-Encoder.
            3) Train CrossEncoder with ``BinaryCrossEntropyLoss``.
            4) Evaluate with:
                - ``CrossEncoderRerankingEvaluator`` (NDCG@10 etc.)
                - ``CrossEncoderClassificationEvaluator`` (binary accuracy, etc.)
            5) Push best checkpoint to the Hub.

        Args:
            data: Optional extra labeled samples (see :meth:`data_preparation`).
            epochs: Number of training epochs.
            batch_size: Train/eval batch size for the CrossEncoder.
            lr: Learning rate.
            warmup_ratio: Warmup schedule ratio.
            eval_split: Test split fraction.
            output_path: Local training output directory.

        Returns:
            None

        Raises:
            ValueError: If the assembled dataset is empty.
        """
        full_dataset = self.data_preparation(data)
        # FIX: Proper emptiness check (the previous `== []` wouldn't raise)
        if len(full_dataset) == 0:
            raise ValueError("No data was given for cross-encoder finetuning.")

        dataset = full_dataset.shuffle(seed=43).train_test_split(test_size=eval_split)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Mine hard negatives with a Bi-Encoder (CPU-friendly by default)
        embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
        hard_train_dataset = mine_hard_negatives(
            train_dataset,
            embedding_model,
            num_negatives=5,         # negatives per (query, positive)
            margin=0,
            range_min=0,
            range_max=100,
            sampling_strategy="top",
            batch_size=4096,
            output_format="labeled-pair",   # (query, passage, label)
            use_faiss=False,
        )

        loss = BinaryCrossEntropyLoss(self.model)

        # Classification evaluator on the held-out split
        pairs = list(zip(eval_dataset["query"], eval_dataset["response"]))
        labels = eval_dataset["label"]
        dev_evaluator = CrossEncoderClassificationEvaluator(
            sentence_pairs=pairs,
            labels=labels,
            name="sts_dev",
        )

        # Reranking evaluator built from mined hard negatives over eval split
        hard_eval_dataset = mine_hard_negatives(
            eval_dataset,
            embedding_model,
            corpus=full_dataset["response"][:],  # Use the full dataset responses as corpus
            num_negatives=30,
            batch_size=4096,
            include_positives=True,
            output_format="n-tuple",
            use_faiss=False,
        )

        reranking_evaluator = CrossEncoderRerankingEvaluator(
            samples=[
                {
                    "query": sample["query"],
                    "positive": [sample["response"]],
                    "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
                }
                for sample in hard_eval_dataset
            ],
            batch_size=batch_size,
            name="gooaq-dev",
            always_rerank_positives=False,
        )

        # Optional: quick pre-train reranking score (kept for visibility)
        _ = reranking_evaluator(self.model)

        evaluator = SequentialEvaluator([reranking_evaluator, dev_evaluator])

        args = CrossEncoderTrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            fp16=False,
            bf16=True,
            dataloader_num_workers=4,
            load_best_model_at_end=True,
            metric_for_best_model="eval_gooaq-dev_ndcg@10",
            eval_strategy="steps",
            eval_steps=4000,
            save_strategy="steps",
            save_steps=4000,
            save_total_limit=2,
            logging_steps=1000,
            logging_first_step=True,
            seed=12,
        )

        # Create the trainer & start training
        trainer = CrossEncoderTrainer(
            model=self.model,
            args=args,
            train_dataset=hard_train_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        trainer.train()

        # Final evaluation (for model card/reporting)
        evaluator(self.model)

        self.push_best_to_hub(self.model, trainer, 'bge-reranker-ft')
