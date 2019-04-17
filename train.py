import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from utils.logging import init_logger
from utils.utils import compute_files_ROUGE


def baseline(args):
    """
    Compute the ROUGE score of oracle summaries.
    """
    dataset = torch.load(os.path.join(args.bert_data_path, 'LCSTS_test.pt'))
    all_src = [f['src_txt'] for f in dataset]
    all_ref = [f['tgt_txt'] for f in dataset]
    all_label = [f['labels'] for f in dataset]
    eval_data = zip(all_src, all_ref, all_label)

    refs = []
    preds = []
    # extract oracle summaries
    for src, ref, label in tqdm(eval_data, total=len(all_src), desc='[Oracle summaries generating] '):
        oracles = [src[i] for i, l in enumerate(label) if l == 1]

        refs.append(ref)
        preds.append(' '.join(oracles))

    # output results
    path_dir = os.path.join(args.result_path, args.mode)
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, 'targets.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(refs))
    with open(os.path.join(path_dir, 'preds.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds))

    compute_files_ROUGE(args, os.path.join(path_dir, 'targets.txt'), os.path.join(path_dir, 'preds.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'oracle'])
    parser.add_argument("-bert_data_path", default='bert_data/')
    parser.add_argument("-model_path", default='models/')
    parser.add_argument("-result_path", default='results/')

    parser.add_argument('-log_file', default='')

    args = parser.parse_args()
    init_logger(args.log_file)

    if args.mode == 'oracle':
        baseline(args)