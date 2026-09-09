import os
import logging
import pandas as pd
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session
from scraper.models import init_db, get_session, JobPosting
from nlp.features import (
    genericness_score,
    vagueness_score,
    repost_score,
    urgency_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_training_set(
    db_path: str = "data/ghostjobs.db",
    output_csv: str = "data/bootstrap_training_set.csv",
) -> str:
    """Build a labeled training set by applying rule-based feature scores to all postings.

    ================================================================================
    BOOTSTRAP HEURISTIC DOCUMENTATION:
    --------------------------------------------------------------------------------
    Postings are auto-labeled as likely-ghost (bootstrap_ghost_label = 1) if the
    average of (genericness_score + vagueness_score + urgency_score) > 0.70.
    This heuristic acts as a silver-standard bootstrap dataset exported to CSV for
    human reviewers to manually verify and correct before fine-tuning DistilBERT.
    ================================================================================
    """
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        postings = session.query(JobPosting).all()
        logger.info(f"Building training set from {len(postings)} JobPosting rows in database.")

        records = []
        for p in postings:
            desc = p.cleaned_text or p.description or ""
            g_score = genericness_score(desc)
            v_score = vagueness_score(desc)
            r_score = repost_score(p)
            u_score = urgency_score(desc)

            feature_avg = (g_score + v_score + u_score) / 3.0
            bootstrap_label = 1 if feature_avg > 0.70 else 0

            records.append({
                "id": p.id,
                "company": p.company,
                "title": p.title,
                "description": desc,
                "genericness_score": g_score,
                "vagueness_score": v_score,
                "repost_score": r_score,
                "urgency_score": u_score,
                "bootstrap_ghost_label": bootstrap_label,
                "manually_verified_label": bootstrap_label,  # Placeholder for human review
            })

        df = pd.DataFrame(records)

        # Ensure parent directory exists
        out_dir = os.path.dirname(output_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"Exported {len(df)} bootstrap training records to '{output_csv}'")
        return output_csv
    finally:
        session.close()


def train_classifier(
    data_csv: str = "data/manually_verified_training_set.csv",
    model_output_dir: str = "models/ghost_bert",
):
    """Fine-tune distilbert-base-uncased model on manually verified labeled dataset.

    Note: This step is blocked on manual review of 'data/bootstrap_training_set.csv'.
    """
    if not os.path.exists(data_csv):
        logger.warning(
            f"Training CSV '{data_csv}' not found. "
            f"Training DistilBERT is blocked until 'data/bootstrap_training_set.csv' is manually verified. "
            f"Pipeline will use weighted feature average fallback in the interim."
        )
        return False

    logger.info(f"Loading training data from '{data_csv}' for DistilBERT fine-tuning...")
    df = pd.read_csv(data_csv)

    if "description" not in df.columns or "manually_verified_label" not in df.columns:
        logger.error(f"Required columns missing in {data_csv}")
        return False

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset as TorchDataset

    class GhostDataset(TorchDataset):
        def __init__(self, texts, labels, tokenizer, max_len=256):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    texts = df["description"].fillna("").tolist()
    labels = df["manually_verified_label"].astype(int).tolist()

    dataset = GhostDataset(texts, labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting DistilBERT fine-tuning...")
    trainer.train()
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logger.info(f"DistilBERT model saved successfully to '{model_output_dir}'")
    return True


def predict(text: str, posting: Any = None, model_dir: str = "models/ghost_bert") -> float:
    """Predict ghost probability score (0.0 to 1.0).

    Uses fine-tuned DistilBERT model if available; falls back to weighted average of rule-based features.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    # 1. Try loading fine-tuned model if directory exists
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.eval()

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                ghost_prob = float(probs[0][1].item())
                return min(1.0, max(0.0, round(ghost_prob, 4)))
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model from '{model_dir}': {e}. Falling back to rule-based features.")

    # 2. Fallback: Weighted average of 4 rule-based feature scores
    g_score = genericness_score(text)
    v_score = vagueness_score(text)
    r_score = repost_score(posting)
    u_score = urgency_score(text)

    weighted_score = (0.30 * g_score) + (0.30 * v_score) + (0.20 * r_score) + (0.20 * u_score)
    return min(1.0, max(0.0, round(float(weighted_score), 4)))
