import json
import argparse
import os
import time
import torch
import numpy as np

from scipy.spatial.distance import cosine
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

import chart_studio
import chart_studio.plotly as py

import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff


chart_studio.tools.set_credentials_file(username=
    'jxhe', api_key='Bm0QOgX4fQf3bULtkpzZ')
chart_studio.tools.set_config_file(world_readable=True,
                             sharing='public')

def read_input(keys, vals):
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

    val_data = []
    with open(vals, 'r') as fin:
        for line in fin:
            s = json.loads(line.strip())
            if s[0] != ' ' and s != '\n':
                s = f'@{s}'
            val_data.append(s)

    return keys, val_data


def plot_html(keys, 
              vals, 
              start_id, 
              length, 
              output_dir, 
              fid='', 
              vis_feat='l2',
              extra=None):

    vis_size = length
    vis_shape = (vis_size // 20, 20)
    num_vis = vis_shape[0] * vis_shape[1]

    symbol = np.reshape(vals[:num_vis], vis_shape)
    keys = keys[:num_vis]

    def cosine_dist(keys):
        dist = [cosine(keys[1], keys[0])]
        for i in range(1, num_vis):
            dist.append(cosine(keys[i], keys[i-1]))

        return np.array(dist)

    def l2_dist(keys):
        diff = [keys[1] - keys[0]]
        for i in range(1, num_vis):
            diff.append(keys[i] - keys[i-1])

        diff = np.array(diff)

        return np.linalg.norm(diff, axis=-1)

    def hyper_z(keys):
        index = extra['model'].coef_[0].nonzero()[0]
        print(f'{index.shape[0]} neurons')

        w = extra['model'].coef_[0][index]
        x = keys[:, index]

        b = extra['model'].intercept_[0]

        # the distance between x0 and hyperplane wx+b=0 is
        # |wx0+b| / |w|
        return (np.dot(x, w) + b) / np.sqrt(np.dot(w, w))


    # import pdb; pdb.set_trace()
    if vis_feat == 'hyper_z':
        hyper_z = hyper_z(keys)
    else:
        hyper_z = None

    l2_d = l2_dist(keys)
    cosine_d = cosine_dist(keys)
    norm = np.linalg.norm(keys, axis=-1)
    shift_norm = np.zeros(norm.shape)
    shift_norm[1:] = norm[:-1]
    shift_norm[0] = shift_norm[1]
    relative_l2 = l2_d / shift_norm

    if vis_feat == 'hyper_z':
        hyper_z = np.reshape(hyper_z, vis_shape)
    else:
        hyper_z = None
    l2_d = np.reshape(l2_d, vis_shape)
    cosine_d = np.reshape(cosine_d, vis_shape)
    norm = np.reshape(norm, vis_shape)
    relative_l2 = np.reshape(relative_l2, vis_shape)


    # Display element name and atomic mass on hover
    hover=[]
    for i in range(vis_shape[0]):
        local = []
        for j in range(vis_shape[1]):
            text = ''
            text += f'norm: {norm[i, j]:.3f} <br>'
            text += f'l2_d: {l2_d[i, j]:.3f} <br>'
            text += f'relative_l2_d: {relative_l2[i, j]:.3f} <br>'
            text += f'cosine_d: {cosine_d[i, j]:.3f} <br>'
            text += f'hyper_z: {hyper_z[i, j]:.3f} <br>' if hyper_z is not None else 'none'

            local.append(text)

        hover.append(local)

    # Invert Matrices
    symbol = symbol[::-1]
    hover = hover[::-1]
    
    plot_args = {'colorscale': 'inferno'}

    if vis_feat == 'l2':
        z = l2_d
    elif vis_feat == 'cosine':
        z = cosine_d
        z[z<0.15] = 0.15
    elif vis_feat == 'norm':
        z = norm
    elif vis_feat == 'relative_l2':
        z = relative_l2
        z[z<0.1] = 1000
        z[z==1000] = 1
    elif vis_feat == 'hyper_z':
        z = hyper_z
        # z = z.flatten()
        z = np.clip(z, -2.5, 2.5)
        plot_args = {'colorscale': 'RdYlGn', 'font_colors': ['black']}
    else:
        raise ValueError

    z = z[::-1]

    # Set Colorscale
    # colorscale=[[0.0, 'rgb(255,255,255)'], [.2, 'rgb(255, 255, 153)'],
    #             [.4, 'rgb(153, 255, 204)'], [.6, 'rgb(179, 217, 255)'],
    #             [.8, 'rgb(240, 179, 255)'],[1.0, 'rgb(255, 77, 148)']]

    # Make Annotated Heatmap
    fig = ff.create_annotated_heatmap(z, annotation_text=symbol, text=hover,
                                     hoverinfo='text', **plot_args)
    fig.update_layout(title_text=f'gpt2-large visualizing {vis_feat} from {fid}')

    fid = fid.split('/')[-1]
    fid = '.'.join(fid.split('.')[:-1])

    if output_dir:
        pio.write_html(fig, file=os.path.join(output_dir, f'gpt2_vis_{vis_feat}_{fid}.html'))
    else:
        py.plot(fig, filename=f'gpt2_vis_{vis_feat}_{fid}.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vis-feat', type=str, default='l2', 
        choices=['l2', 'cosine', 'hyper_z'],
        help='the feature used to reflect colors of the heatmap')
    parser.add_argument('--key', type=str, default='features.jsonl',
        help='the input jsonl file')
    parser.add_argument('--val', type=str, default='features.jsonl',
        help='the input jsonl file')
    parser.add_argument('--max-tok', type=int, default=1e5,
        help='maximum number of tokens to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
        help='the output html directory. If not set, the figures would \
        be uploaded to plotly chart studio')
    parser.add_argument('--extra-path', type=str, default=None,
        help='some extra files to load for visualization, depending \
        on the specific mode of vis_feat')
    # parser.add_argument('--prefix', type=str, default='',
    #     help='the prefix of outputs files, to distinguish in case')
    args = parser.parse_args()

    np.random.seed(22)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    print('reading features')
    start = time.time()

    keys, vals = read_input(args.key, args.val)

    print(f'reading features complete, costing {time.time() - start} seconds')

    length = min(len(keys), args.max_tok)
    keys = keys[:length]
    vals = vals[:length]

    extra = torch.load(args.extra_path) if args.extra_path is not None else None

    plot_html(keys, vals, 0, length, args.output_dir,
        vis_feat=args.vis_feat, fid=args.key, extra=extra)
