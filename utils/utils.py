import os
from rouge import Rouge


def compute_files_ROUGE(args, ref_dir, pred_dir):
    """
    Compute the ROUGE between two files.
    """
    r = Rouge()

    # read files
    with open(ref_dir, 'r', encoding='utf-8') as f:
        refs = f.readlines()
    with open(pred_dir, 'r', encoding='utf-8') as f:
        preds = f.readlines()
    refs = [' '.join(r) for r in refs]
    preds = [' '.join(p) for p in preds]

    rouge_score = r.get_scores(refs, preds)
    print(f'[{args.mode} ROUGE score]')
    print(">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        rouge_score[0]["rouge-1"]["f"] * 100,
        rouge_score[0]["rouge-2"]["f"] * 100,
        rouge_score[0]["rouge-l"]["f"] * 100,
        rouge_score[0]["rouge-1"]["r"] * 100,
        rouge_score[0]["rouge-2"]["r"] * 100,
        rouge_score[0]["rouge-l"]["r"] * 100
    ))

    # output to files
    with open(os.path.join(args.result_path, args.mode, 'rouge.txt'), 'w', encoding='utf-8') as f:
        f.write("ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
            rouge_score[0]["rouge-1"]["f"] * 100,
            rouge_score[0]["rouge-2"]["f"] * 100,
            rouge_score[0]["rouge-l"]["f"] * 100,
            rouge_score[0]["rouge-1"]["r"] * 100,
            rouge_score[0]["rouge-2"]["r"] * 100,
            rouge_score[0]["rouge-l"]["r"] * 100
        ))
