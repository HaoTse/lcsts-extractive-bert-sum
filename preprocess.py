import argparse

from utils.logging import init_logger
from prepro import data_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-mode", nargs='+', default='', type=str, help='tokenize or format_to_bert')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')
    parser.add_argument("-save_temp", type=str2bool, nargs='?', const=False, default=False, help='if save the tokenized result (default is False).')

    # parser.add_argument("-train_src_path", default='data/train-large/short_text_t.txt')
    # parser.add_argument("-train_tgt_path", default='data/train-large/summary_t.txt')
    # parser.add_argument("-valid_src_path", default='data/train-small/short_text_t.txt')
    # parser.add_argument("-valid_tgt_path", default='data/train-small/summary_t.txt')
    # parser.add_argument("-test_src_path", default='data/test/short_text_t.txt')
    # parser.add_argument("-test_tgt_path", default='data/test/summary_t.txt')

    parser.add_argument("-train_src_path", default='small_data/train/short_text.txt')
    parser.add_argument("-train_tgt_path", default='small_data/train/summary.txt')
    parser.add_argument("-valid_src_path", default='small_data/train/short_text.txt')
    parser.add_argument("-valid_tgt_path", default='small_data/train/summary.txt')
    parser.add_argument("-test_src_path", default='small_data/test/short_text.txt')
    parser.add_argument("-test_tgt_path", default='small_data/test/summary.txt')

    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=0, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('-log_file', default='')

    parser.add_argument('-dataset', nargs='+', default=['train', 'valid', 'test'], help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    result = data_builder.tokenize(args)
    data_builder.format_to_bert(args, result)