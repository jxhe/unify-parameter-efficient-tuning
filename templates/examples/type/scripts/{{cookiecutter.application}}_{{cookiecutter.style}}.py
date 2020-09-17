from transformers import AutoTokenizer, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, {{cookiecutter.model_class_name}}
from datasets import load_dataset, load_metric, temp_seed
from torch.utils.data import DataLoader
import argparse
import os

MAX_GPU_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

def prepare_dataset(args):
    """ Load a dataset, a tokenizer and encode the dataset """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(args.dataset_name, args.dataset_config_name)
    # If you have a custom dataset in a CSV/JSON/TEXT file
    # dataset = load_dataset("csv", data_files={'train': "train_data.csv", 'validation': "valid_data.csv"})

    # Prepare the dataset (cached)
    def encode(example):
        output = tokenizer(example['sentence1'], example['sentence2'], truncation=True)
        output['labels'] = example['label']
        return output

    dataset = dataset.map(encode)
    dataset.set_format(columns=['attention_mask', 'input_ids', 'token_type_ids', 'labels'])

    return dataset, tokenizer


def train_and_evaluate(args):
    # Load dataset, tokenizer and metric
    dataset, tokenizer = prepare_dataset(args)
    metric = load_metric(args.dataset_name, args.dataset_config_name)

    # Instantiate train and evaluation dataloaders
    def collate_fn(examples):
        """ Collate function which pads the batches to the longest sequence in the batch and returns PyTorch tensors """
        return tokenizer.pad(examples, padding='longest', return_tensors='pt')

    train_dataloader = DataLoader(dataset['train'], shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(dataset['validation'], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE)

    # Instantiate model
    model = {{cookiecutter.model_class_name}}.from_pretrained(args.model_name_or_path, return_dict=True)

    # Instantiate optimizer and learning rate schedule
    optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=args.correct_bias)
    if args.lr_schedule == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader)*args.num_epochs)

    # Train the model
    eval_metrics = {}
    for epoch in range(args.num_epochs):
        model.train()
        model.to(args.device)
        for step, batch in enumerate(train_dataloader):
            batch.to(args.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch.to(args.device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

        eval_metrics = metric.compute()
        print(f"eval_metric for epoch {epoch}:", eval_metrics)

    return eval_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir", default=False, type=bool, help="Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Run evaluation during training at each logging step.")
    parser.add_argument("--prediction_loss_only", action='store_true', help="When performing evaluation and predictions, only returns the loss.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_dir", type=str, default_factory=default_logdir, help="Tensorboard log dir.")
    parser.add_argument("--logging_first_step", action='store_true', help="Log and eval the first global_step")
    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None, help="Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--dataloader_drop_last", action='store_true', help="Drop the last incomplete batch if it is not divisible by the batch size.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run an evaluation every X steps.")
    parser.add_argument("--past_index", type=int, default=-1, help="If >=0, uses the corresponding part of the output as the past state for next step.")
    parser.add_argument("--run_name", type=str, default=None, help="An optional descriptor for the run. Notably used for wandb logging.")
    parser.add_argument("--disable_tqdm", action='store_true', help="Whether or not to disable the tqdm progress bars.")
    args = parser.parse_args()

    with temp_seed(args.seed, set_pytorch=True):
        return train_and_evaluate(args)

if __name__ == "__main__":
    main()
