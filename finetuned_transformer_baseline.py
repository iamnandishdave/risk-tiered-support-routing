"""Fine-tuned transformer baseline (e.g., BERT/RoBERTa) — NOT YET RUN.
This sandbox has no outbound access to huggingface.co, so pretrained weights could not
be downloaded here. Run this on a machine with internet access and, ideally, a GPU.
pip install transformers torch datasets accelerate"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MODEL_NAME = "bert-base-uncased"  # or "roberta-base"

def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    from datasets import Dataset

    tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
    le = LabelEncoder().fit(tr.category)
    tr['label'] = le.transform(tr.category); te['label'] = le.transform(te.category)
    tr_fit, val = train_test_split(tr, test_size=0.2, random_state=42, stratify=tr.category)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def enc(df): return Dataset.from_pandas(df[['text','label']]).map(
        lambda b: tok(b['text'], truncation=True, padding='max_length', max_length=48), batched=True)
    ds_tr, ds_val, ds_te = enc(tr_fit), enc(val), enc(te)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))
    args = TrainingArguments(output_dir='bert_out', num_train_epochs=4, per_device_train_batch_size=32,
                              per_device_eval_batch_size=64, eval_strategy='epoch', save_strategy='no',
                              learning_rate=2e-5, seed=42, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_val)
    trainer.train()
    preds = trainer.predict(ds_te)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    import numpy as np
    pred_idx = probs.argmax(1); conf = probs.max(1)
    out = pd.DataFrame({'text': te.text, 'true_category': te.category,
                         'bert_intent': le.inverse_transform(pred_idx), 'bert_confidence': conf})
    out.to_csv('results_bert_classifier.csv', index=False)
    print('Accuracy:', (out.bert_intent==out.true_category).mean())
    print('Feed results_bert_classifier.csv through the same route()/metrics() functions in analysis2.py.')

if __name__ == "__main__":
    main()
