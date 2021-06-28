"""this script encodes the given text piece,
potentially allowing for different hidden
representations
"""

import os
import json
import shutil
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, GPT2TokenizerFast
from openai_sentiment_neuron import sst_binary

from datasets import load_dataset


def encode(model, x, save_dir, split, device, offset=0, bsz=32):
    hidden_type = 'standard' if args.return_hidden_type is None else args.return_hidden_type
    ids = f'layer{args.nlayer}.rpr_type_{hidden_type}.offset_{offset}'

    keys = np.memmap(os.path.join(save_dir, f'{split}.keys.{ids}.size{len(x)}.hid{model.config.n_embd}.npy'),
                     dtype=np.float32,
                     mode='w+',
                     shape=(len(x), model.config.n_embd))

    max_length = model.config.n_positions

    lls = []
    cur = 0
    total_tok = 0

    pbar = tqdm(total=len(x))

    # import pdb; pdb.set_trace()

    while cur < len(x):
        cur_bsz = min(bsz, len(x) - cur)
        xi = x[cur:cur+cur_bsz]
        encodings = tokenizer(xi, padding=True, 
            truncation=True, return_tensors='pt',
            max_length=max_length).to(device)

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        target_ids = input_ids.clone()

        trg_len = attention_mask.sum(1)
        for i in range(target_ids.size(0)):
            target_ids[i, trg_len[i]:] = -100

        # the perplexity computation is not very accurate
        # since it ignores the first token prediction and eos prediction

        with torch.no_grad():
            # labels are shifted inside the model
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids,
                output_hidden_states=True, return_hidden_type=args.return_hidden_type)
            log_likelihood = outputs[0] * ((trg_len-1).sum())
            total_tok += (trg_len-1).sum().item()
            hidden_states = outputs.hidden_states

            # import pdb; pdb.set_trace()
            hidden_states = hidden_states[args.nlayer]

            for i in range(cur_bsz):
                keys[cur+i] = hidden_states[i, trg_len[i].item()-1 + offset, :].cpu().numpy().astype(np.float32)
                # keys[cur+i] = hidden_states[i, 1, :].cpu().numpy().astype(np.float32)

        cur += cur_bsz
        pbar.update(cur_bsz)

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / total_tok)
    print(f'split perplexity {ppl}')



parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['sst', 'imdb'],
    help='the data directory which consists of csv files')
parser.add_argument('--nlayer', type=int, default=-1,
    help='the order of layer from which we extract hidden states, \
    default to the last layer')
parser.add_argument('--save-emb', type=str, default='hidden_val',
    help='the folder to save the encodings')
parser.add_argument('--model', type=str, default='gpt2-large',
    help='the pretrained model name')
parser.add_argument('--return-hidden-type', type=str, default=None, \
    choices=['ffn_input_after_ln'], \
    help='the hidden representations to use, by default we use the output of every \
    sub transformer layer')
parser.add_argument('--offset', type=int, default=0, \
    help='the offset of encoding positions in the sequence, by default we use the last hidden state')

args = parser.parse_args()

save_dir = f'{args.save_emb}'

# if os.path.isdir(save_dir):
#     shutil.rmtree(save_dir)


os.makedirs(args.save_emb, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)

tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

if args.dataset == 'sst':
    trX, vaX, teX, trY, vaY, teY = sst_binary('openai_sentiment_neuron/data')
    bsz = 32
    data = {'train': trX, 'val':vaX, 'test': teX}
elif args.dataset == 'imdb':
    bsz = 2
    dataset = load_dataset('imdb', cache_dir='data/hf_data_cache')
    trX = [x['text'] for x in dataset['train']]
    teX = [x['text'] for x in dataset['test']]
    data = {'train': trX, 'test': teX}

for split, x in data.items():
    encode(model, x, save_dir, split, device, 
        offset=args.offset, bsz=bsz)

