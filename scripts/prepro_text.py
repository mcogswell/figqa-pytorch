#!/usr/bin/env python
import os
import os.path as pth
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from PIL import Image

from figqa.utils.dataset import ques_to_tensor

def tokenize_qas(qa_pairs):
    for qa_pair in qa_pairs:
        qa_pair['question'] = word_tokenize(qa_pair['question_string'])
        if 'UNK' in qa_pair['question']:
            print(qa_pair['question'], qa_pair['question_string'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Process figureqa text into tensors: '
                'Extract vocab, answer tensor, question tensor')

    # Input files
    parser.add_argument('--figqa-dir',
        default='data/figureqa/',
        help='directory containing unzipped figqa files')
    # NOTE: These should be processed at once because the complete
    # set determines the vocab.
    splits = ['no_annot_test1', 'no_annot_test2', 'validation1',
              'validation2', 'train1', 'sample_train1']
    # Output files
    parser.add_argument('--output-dir',
        default='data/figureqa_pre/',
        help='Save one hdf5 per dataset here')
    # Options
    parser.add_argument('--max_ques_len', default=40, type=int,
        help='Max length of questions')
    args = parser.parse_args()

    # 1: load QA pairs and extract vocab
    qa_pairs = {}
    vocab = set()
    for split in splits:
        with open(pth.join(args.figqa_dir, split, 'qa_pairs.json'), 'r') as f:
            split_qa_pairs = json.load(f)['qa_pairs']
        tokenize_qas(split_qa_pairs)
        qa_pairs[split] = split_qa_pairs
        for qap in split_qa_pairs:
            vocab.update(qap['question'])
    vocab = sorted(list(vocab))
    vocab = ['NULL', '<START>', '<END>'] + vocab

    # 2: save vocab
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_fname = pth.join(args.output_dir, 'vocab.json')
    word2ind = {w: i for i, w in enumerate(vocab)}
    with open(vocab_fname, 'w') as f:   
        json.dump({
            'ind2word': vocab, # just a list
            'word2ind': word2ind
        }, f)

    # 3: save questions and answers as tensors (without start and end tokens)
    for split in splits:
        os.makedirs(pth.join(args.output_dir, split), exist_ok=True)
        fname = pth.join(args.output_dir, split, 'qa_pairs.h5')
        f = h5py.File(fname, 'w')

        # questions -> tensor
        questions = (qap['question'] for qap in qa_pairs[split])
        questions = [ques_to_tensor(q, word2ind) for q in questions]
        questions = np.stack(questions)
        f.create_dataset('questions', dtype='uint32', data=questions)
        # answers -> tensor (0 or 1 for no or yes)
        if split not in ['no_annot_test1', 'no_annot_test2']:
            answers = [qap['answer'] for qap in qa_pairs[split]]
            answers = np.array(answers, dtype='uint32')
            f.create_dataset('answers', dtype='uint32', data=answers)
        # image indices
        image_idx = [qap['image_index'] for qap in qa_pairs[split]]
        image_idx = np.array(image_idx, dtype='uint32')
        f.create_dataset('image_idx', dtype='uint32', data=image_idx)

        f.close()

