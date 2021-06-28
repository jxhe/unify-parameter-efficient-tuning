import argparse
import numpy as np
from openai_sentiment_neuron import sst_binary, train_with_reg_cv, train_with_reg


def read_input(keys):
    def parse_fname(fname):
        x = '.'.join(fname.split('.')[:-1])
        x = x.split('/')[-1]
        x = x.split('.')

        size = int(x[-2].split('size')[-1])
        embed = int(x[-1].split('hid')[-1])

        return size, embed

    size, embed = parse_fname(keys)
    keys = np.memmap(keys,
                     dtype=np.float32,
                     mode='r',
                     shape=(size, embed))

    return keys

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='openai_sentiment_neuron/data',
    help='the data directory which consists of csv files')
parser.add_argument('--train', type=str,
    help='the data directory which consists of csv files')
parser.add_argument('--val', type=str,
    help='the data directory which consists of csv files')
parser.add_argument('--test', type=str,
    help='the data directory which consists of csv files')
parser.add_argument('--c', type=float, default=1,
    help='inverse the regularization constant')

args = parser.parse_args()

trX, vaX, teX, trY, vaY, teY = sst_binary(args.data)

trXt = read_input(args.train)
vaXt = read_input(args.val)
teXt = read_input(args.test)

# classification results
full_rep_acc, c, nnotzero, model = train_with_reg(trXt, trY, vaXt, vaY, teXt, teY, c=args.c, verbose=1)
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)
