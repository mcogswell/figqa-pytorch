import os.path as pth
import ujson as json
import numpy as np
import h5py
from PIL import Image
import PIL.ImageOps as ImageOps

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from figqa.utils.sequences import NULL

class FigQADataset(Dataset):

    def __init__(self, dname, prepro_dname, split):
        '''
        PyTorch Dataset for loading FigureQA data from pre-processed
        h5 files (see scripts/prepro_text.py). Questions, answers, images,
        and some meta-data are loaded.

        Arguments:
            dname: directory of raw FigureQA download (one directory per split)
            prepro_dname: directory with preprocessed h5 files
            split: name of dataset split (e.g., train1, validation2, ...)
        '''
        self.dname = dname
        self.prepro_dname = prepro_dname
        self.split = split

        # load QAs into numpy arrays
        fname = pth.join(prepro_dname, split, 'qa_pairs.h5')
        self.qa_pairs = h5py.File(fname)
        with open(pth.join(dname, split, 'qa_pairs.json'), 'r') as f:
            self.qa_pairs_json = json.load(f)['qa_pairs']
        self.questions = np.array(self.qa_pairs['questions']).astype('int')
        self.answers = np.array(self.qa_pairs['answers']).astype('int')
        self.image_idx = np.array(self.qa_pairs['image_idx'])

        # image->tensor transform
        self.transform = transforms.Compose([
                            transforms.Lambda(self.resize),
                            transforms.Lambda(self.pad),
                            transforms.RandomCrop(256, padding=8),
                            transforms.ToTensor(),
                        ])

    @staticmethod
    def resize(img):
        '''Resize img so the largest dimension is 256'''
        msize = max(img.size)
        height = int(256 * img.size[0] / msize)
        width = int(256 * img.size[1] / msize)
        return img.resize((height, width))

    @staticmethod
    def pad(img):
        '''Make an image 256x256 by padding with 0s'''
        msize = min(img.size)
        if msize == 256:
            return img
        height, width = img.size
        pad1 = (256 - msize) // 2
        pad2 = (256 - msize) - pad1
        left, top, right, bottom = 0, 0, 0, 0
        if height > width:
            top, bottom = pad1, pad2
        else:
            left, right = pad1, pad2
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
        assert img.size == (256, 256)
        return img

    def __getitem__(self, index):
        # question-answer info
        question = self.questions[index]
        answer = self.answers[index]
        image_idx = self.image_idx[index]
        question_len = (question == NULL).nonzero()[0].min()
        qtype = self.qa_pairs_json[index]['question_id']

        # load image
        fname = '{}.png'.format(image_idx)
        path = pth.join(self.dname, self.split, 'png', fname)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        return {
            'img': img,
            'question': question,
            'question_len': question_len,
            'qtype': qtype,
            'answer': answer,
        }

    def __len__(self):
        return len(self.qa_pairs['questions'])
