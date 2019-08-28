import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME

from train import str2bool
from utils.logging import logger, init_logger
from summarizer.model_builder import Summarizer
from prepro.data_builder import BertData

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, docu_idx, seg_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask):
        self.docu_idx = docu_idx
        self.seg_idx = seg_idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.clss_ids = clss_ids
        self.clss_mask = clss_mask

def convert_examples_to_features(insts):
    """Loads a data file into a list of `InputBatch`s."""

    max_seq_length = max(max([len(e['src']) for e in examples]) for examples in insts)
    max_cls_lenght = max(max([len(e['clss']) for e in examples]) for examples in insts)

    features = []
    for (docu_index, examples) in enumerate(insts):
        for (seg_index, example) in enumerate(examples):
            input_ids = example['src']
            segment_ids = example['segs']
            clss_ids = example['clss']

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            clss_mask = [1] * len(clss_ids)
            clss_padding = [0] * (max_cls_lenght - len(clss_ids))
            clss_ids += clss_padding
            clss_mask += clss_padding

            assert len(clss_ids) == max_cls_lenght
            assert len(clss_mask) == max_cls_lenght

            features.append(
                    InputFeatures(docu_idx=docu_index,
                                seg_idx=seg_index,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                clss_ids=clss_ids,
                                clss_mask=clss_mask))
    return features

def get_dataset(features):
    """Pack the features into dataset"""
    all_docu_idx = torch.tensor([f.docu_idx for f in features], dtype=torch.int)
    all_seg_idx = torch.tensor([f.seg_idx for f in features], dtype=torch.int)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_clss_ids = torch.tensor([f.clss_ids for f in features], dtype=torch.long)
    all_clss_mask = torch.tensor([f.clss_mask for f in features], dtype=torch.long)
    
    return TensorDataset(all_docu_idx, all_seg_idx, all_input_ids, all_input_mask, all_segment_ids, all_clss_ids, all_clss_mask)

def select_sent(docu_idx, seg_idx, examples, scores, mask):
    """Find the prediction."""
    sent_scores = scores + mask
    selected_ids = np.argsort(-sent_scores, 1) # [batch size, clss length, 1]

    result = []
    for i, idx in enumerate(selected_ids):
        _pred = []

        # get example data
        example = examples[docu_idx[i]][seg_idx[i]]
        src_str = example['src_txt']

        _pred = src_str[selected_ids[i][0]].strip()

        result.append((docu_idx[i], _pred))
    
    return result

def get_data(args):
    line = '2007年喬布斯向人們展示iPhone並宣稱“它將會改變世界” 還有人認爲他在誇大其詞 然而在8年後 以iPhone爲代表的觸屏智能手機已經席捲全球各個角落<sep>未來 智能手機將會成爲“真正的個人電腦” 爲人類發展做出更大的貢獻'
    bert = BertData(args)
    
    pt_result = []

    segs = line.split('<sep>')
    segs = [seg.strip() for seg in segs if seg.strip()]
    seg_result = []
    for seg in segs:
        sents = seg.split(' ')
        indexed_tokens, segments_ids, cls_ids, src_txt = bert.convert_format(sents)
        b_data_dict = {"src": indexed_tokens, "segs": segments_ids, 'clss': cls_ids, 'src_txt': src_txt}
        seg_result.append(b_data_dict)
    pt_result.append(seg_result)

    return pt_result

def sum_file(args):
    # initial device and gpu number
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_file = os.path.join(args.model_path, WEIGHTS_NAME)
    config_file = os.path.join(args.model_path, CONFIG_NAME)
    # Prepare model
    print('[Test] Load model...')
    config = BertConfig.from_json_file(config_file)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.to(device)
    # load check points
    model.load_cp(torch.load(model_file))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare testing data
    eval_dataloader = None

    eval_examples = get_data(args)
    eval_features = convert_examples_to_features(eval_examples)
    logger.info("  Num documents = %d", len(eval_examples))
    logger.info("  Num segments = %d", len(eval_features))
    
    # Run prediction for full data
    eval_data = get_dataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    # testing
    preds = [[] for _ in range(len(eval_examples))]
    if eval_dataloader:
        model.eval()
        for docu_idx, seg_idx, input_ids, input_mask, segment_ids, clss_ids, clss_mask in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            clss_ids = clss_ids.to(device)
            clss_mask = clss_mask.to(device)

            with torch.no_grad():
                sent_scores, mask = model(input_ids, segment_ids, clss_ids, input_mask, clss_mask)

            sent_scores = sent_scores.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            # find selected sentences
            result = select_sent(docu_idx, seg_idx, eval_examples, sent_scores, mask)
            for i, pred in result:
                preds[i].append(pred)
    preds = ['<sep>'.join(pred) for pred in preds]

    # output results
    print('\n'.join(preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)

    parser.add_argument('--max_nsents', default=100, type=int)
    parser.add_argument('--min_src_ntokens', default=0, type=int)
    parser.add_argument('--max_src_ntokens', default=200, type=int)

    parser.add_argument("--bert_model", default='bert-base-chinese', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--param_init", default=0, type=float)
    parser.add_argument("--param_init_glorot", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    init_logger()

    sum_file(args)