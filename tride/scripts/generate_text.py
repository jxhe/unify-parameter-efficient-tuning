"""sample text from the pretrained language model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

parser = argparse.ArgumentParser()
# parser.add_argument('--prompt', type=str, default='',
    # help='the prompt to start with')
parser.add_argument('--model', type=str, default='gpt2-large',
    help='the pretrained model name')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model)

model.to(device)
model.eval()

prompt="In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

# encode input context
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

outputs = model.generate(input_ids=None if prompt=='' else input_ids, do_sample=True, max_length=512, top_k=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
