import os
import os.path as pth
import pickle as pkl
import shutil
from timeit import default_timer as timer
from pprint import pprint
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jinja2 import Environment, FileSystemLoader, select_autoescape

import figqa.options
import figqa.utils as utils
from figqa.utils.datasets import FigQADataset, batch_iter, ques_tensor_to_str


def render_webpage(batches, preds, out_dir, vocab):
    env = Environment(
        loader=FileSystemLoader('vis_static'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    out_img_dir = pth.join(out_dir, 'images')
    os.makedirs(out_img_dir, exist_ok=True)

    examples = []
    for batch, pred, in zip(batches, preds):
        gt_answers = ['yes' if ans else 'no' for ans in batch['answer']]
        _, pred_answers = pred.max(dim=1)
        pred_answers = ['yes' if ans else 'no' for ans in pred_answers]
        for idx in range(len(pred)):
            img_fname = pth.basename(batch['img_path'][idx])
            out_img_path = pth.join(out_img_dir, img_fname)
            if not pth.exists(out_img_path):
                shutil.copy(batch['img_path'][idx], out_img_path)
            question = ques_tensor_to_str(batch['question'][idx], vocab['ind2word'])

            examples.append({
                'img': 'images/' + img_fname,
                'question': question,
                'gt_answer': gt_answers[idx],
                'pred_answer': pred_answers[idx],
            })

    template = env.get_template('main.html')
    return template.render(examples=examples)

def main(args):
    # data
    split = args.val_split
    dataset = FigQADataset(args.figqa_dir, args.figqa_pre,
                            split=split, max_examples=args.max_examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.workers)

    # model
    model, model_args = utils.load_model(fname=args.start_from,
                                         return_args=True,
                                         ngpus=args.cuda)
    model.eval()
    criterion = nn.NLLLoss()

    # evaluate metrics on dataset
    preds = []
    batches = []
    for batch_idx, batch in batch_iter(dataloader, args, volatile=True):
        if batch_idx % 50 == 0:
            print('Batch {}/{}'.format(batch_idx, len(dataloader)))
        # forward
        pred = model(batch)
        loss = criterion(pred, batch['answer'])

        # visualization
        preds.append(pred)
        batches.append(batch)

    # save webpage that displays example predictions
    with open(pth.join(args.figqa_pre, 'vocab.json'), 'r') as f:
        vocab = json.load(f)
    html = render_webpage(batches, preds, args.examples_dir, vocab)
    with open(pth.join(args.examples_dir, 'examples.html'), 'w') as f:
        f.write(html)


if __name__ == '__main__':
    main(figqa.options.parse_arguments())
