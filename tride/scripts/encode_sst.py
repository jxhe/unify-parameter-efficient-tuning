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
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from openai_sentiment_neuron.utils import sst_binary


def encode(model, x, save_dir, split, device):
    hidden_type = 'standard' if args.return_hidden_type is None else args.return_hidden_type
    ids = f'layer{args.nlayer}.rpr_type_{hidden_type}'

    keys = np.memmap(os.path.join(save_dir, f'{split}.keys.{ids}.size{encodings.input_ids.size(1)}.hid{model.config.n_embd}.npy'),
                     dtype=np.float32,
                     mode='w+',
                     shape=(len(x), model.config.n_embd))

    max_length = model.config.n_positions
    bsz = 

    lls = []
    cur = 0
    bsz = 128
    total_tok = 0

    pbar = tqdm(total=len(x))

    while cur < len(x):
        cur_bsz = min(bsz, len(x) - cur)
        xi = x[cur:cur+cur_bsz]
        encodings = tokenizer(xi, return_tensors='pt')

        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()

        trg_len = 1
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            # labels are shifted inside the model
            outputs = model(input_ids, labels=target_ids,
                output_hidden_states=True, return_hidden_type=args.return_hidden_type)
            log_likelihood = outputs[0] * trg_len
            hidden_states = outputs.hidden_states

            hidden_states = hidden_states[args.nlayer]
            keys[cur:cur+cur_bsz] = hidden_states[0][-1, :].cpu()

        cur += cur_bsz
        pbar.update(cur_bsz)

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / total_tok)
    print(f'split perplexity {ppl}')



parser = argparse.ArgumentParser()
parser.add_argument('data', type=str,
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

args = parser.parse_args()

save_dir = f'{args.save_emb}'

# if os.path.isdir(save_dir):
#     shutil.rmtree(save_dir)


os.makedirs(args.save_emb, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)

model.to(device)
model.eval()

trX, vaX, teX, trY, vaY, teY = sst_binary(args.data)

data = {'train': trX, 'val':vaX, 'test': teX}
for split, x in data.items():
    encode(model, x, save_dir, split, device):
