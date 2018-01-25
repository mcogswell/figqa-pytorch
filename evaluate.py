import os
import os.path as pth
import pickle as pkl
from timeit import default_timer as timer
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import figqa.options
import figqa.utils as utils
from figqa.utils.datasets import FigQADataset

from train import batch_iter


def main(args):
    running_loss = None
    start_t = None

    # data
    split = args.val_split
    dataset = FigQADataset(args.figqa_dir, args.figqa_pre,
                            split=split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.workers)

    # model
    model = utils.load_model(fname=args.start_from)
    if args.cuda:
        model = model.cuda()
    model.eval()
    criterion = nn.NLLLoss()

    # evaluate metrics on dataset
    accs = []
    accs_by_qtype = {qtype: [] for qtype, _ in enumerate(utils.QTYPE_ID_TO_META)}
    start_t = timer()
    for batch_idx, batch in batch_iter(dataloader, args, volatile=True):
        if batch_idx % 50 == 0:
            print('Batch {}/{}'.format(batch_idx, len(dataloader)))
        # forward
        pred = model(batch)
        loss = criterion(pred, batch['answer'])

        # accuracy
        _, pred_idx = torch.max(pred, dim=1)
        correct = (batch['answer'] == pred_idx)
        acc = correct.cpu().data.numpy()
        accs.append(acc)
        for qtype, meta in enumerate(utils.QTYPE_ID_TO_META):
            qtype_mask = (batch['qtype'] == qtype)
            if qtype_mask.sum().data[0] == 0:
                continue
            acc = correct[qtype_mask].cpu().data.numpy()
            accs_by_qtype[qtype].append(acc)

    # accumulate results into convenient dict
    accs = np.concatenate(accs, axis=0)
    for qtype in accs_by_qtype:
        qaccs = accs_by_qtype[qtype]
        accs_by_qtype[qtype] = np.concatenate(qaccs, axis=0).mean()
    result = {
        'split': split,
        'model_kind': model.kind,
        'acc': accs.mean(),
        'accs_by_qtype': accs_by_qtype,
        'qtypes': [qt[0] for qt in utils.QTYPE_ID_TO_META],
    }
    pprint(result)
    result['args'] = args
    result_name = args.result_name

    # save to disk
    name = 'result_{split}_{result_name}.pkl'.format(**locals())
    result_fname = pth.join(args.result_dir, name)
    os.makedirs(args.result_dir, exist_ok=True)
    with open(result_fname, 'wb') as f:
        pkl.dump(result, f)


if __name__ == '__main__':
    main(figqa.options.parse_arguments())
