"""Fine-tuned transformer classifier (BERT/RoBERTa) as an alternative front end.

Trains on the same 80/20 fit/validation split as main_results.py (seed 42) and
saves full intent-probability matrices for the validation and test sets, so the
same threshold selection and P0/P1/P2 routing can be applied.

Requirements: a GPU is recommended.
    pip install transformers torch datasets accelerate

Outputs:
    bert_val_probs.npy   (2,001 x 77)  validation-set probabilities
    bert_test_probs.npy  (3,080 x 77)  test-set probabilities
    bert_classes.csv     intent order of the probability columns
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MODEL_NAME = "bert-base-uncased"  # or "roberta-base"
SEED = 42


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, set_seed)
    set_seed(SEED)

    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    le = LabelEncoder().fit(tr.category)          # alphabetical, same order as sklearn's classes_
    tr['label'] = le.transform(tr.category)
    te['label'] = le.transform(te.category)
    tr_fit, val = train_test_split(tr, test_size=0.20, random_state=SEED,
                                   stratify=tr.category)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def enc(df):
        return Dataset.from_pandas(df[['text', 'label']].reset_index(drop=True)).map(
            lambda b: tok(b['text'], truncation=True, padding='max_length', max_length=48),
            batched=True)

    ds_fit, ds_val, ds_te = enc(tr_fit), enc(val), enc(te)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_))
    args = TrainingArguments(output_dir='bert_out', num_train_epochs=4,
                             per_device_train_batch_size=32, per_device_eval_batch_size=64,
                             eval_strategy='epoch', save_strategy='no',
                             learning_rate=2e-5, seed=SEED, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds_fit, eval_dataset=ds_val)
    trainer.train()

    def probs_for(ds):
        logits = trainer.predict(ds).predictions
        return torch.softmax(torch.tensor(logits), dim=1).numpy()

    p_val, p_test = probs_for(ds_val), probs_for(ds_te)
    np.save('bert_val_probs.npy', p_val)
    np.save('bert_test_probs.npy', p_test)
    pd.Series(le.classes_, name='intent').to_csv('bert_classes.csv', index=False)

    acc = (le.classes_[p_test.argmax(1)] == te.category.values).mean()
    print(f'BERT test accuracy: {acc:.4f}')
    print('Pass bert_val_probs.npy and bert_test_probs.npy to main_results.py '
          'in place of the TF-IDF probabilities (Pv and Pt).')


if __name__ == "__main__":
    main()
