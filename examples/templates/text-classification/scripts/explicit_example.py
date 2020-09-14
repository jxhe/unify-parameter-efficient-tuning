import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, logging

logger = logging.get_logger(__name__)

model_name = 'bert-base-cased'
dataset_name, dataset_subset = 'glue', 'mrpc'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
dataset = load_dataset(dataset_name, dataset_subset)
metric = load_metric(dataset_name, dataset_subset)


# Prepare the dataset: encode it as input to our model and truncate to max model length
def encode(examples):
    output = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)
    output['labels'] = examples['label']
    return output

encoded_dataset = dataset.map(encode, batched=True)
encoded_dataset.set_format(columns=['input_ids', 'token_type_ids', 'labels'])

def train_epoch(model, optimizer, train_dataloader, eval_dataloader, device):
    model.to(device)
    model.train()
    for batch in train_dataloader:
        batch.to(device)
        model(**batch)
        loss = model.losss
        loss.backward()
        optimizer.step()

    model.eval()
    for batch in eval_dataloader:
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
    scores = metric.compute()
    return scores

learning_rate = 1e-5
train_batch_size = 16
seed = 42
epochs = 2

# Prepare pytorch dataloader (with dynamic batch i.e. pad the sequences to the longest in the batch)
def collate_fn(examples):
  return tokenizer.pad(examples, padding='longest', return_tensors='pt')

train_dataloader = DataLoader(dataset['train'], shuffle=True, collate_fn=collate_fn, batch_size=32)
valid_dataloader = DataLoader(dataset['validation'], shuffle=False, collate_fn=collate_fn, batch_size=16)

optimizer = AdamW(params=model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    eval_metrics = train_epoch(model, optimizer, train_dataloader, valid_dataloader)
    print(f"eval_metrics for trial {trial.number} epoch {epoch}:", eval_metrics)
    sum_metrics = sum(eval_metrics.values())
