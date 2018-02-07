import torch
import torch.nn as nn

QTYPE_ID_TO_META = [
    ('Is X the minimum?', ('bar', 'pie')),
    ('Is X the maximum?', ('bar', 'pie')),
    ('Is X the low median?', ('bar', 'pie')),
    ('Is X the high median?', ('bar', 'pie')),
    ('Is X less than Y?', ('bar', 'pie')),
    ('Is X greater than Y?', ('bar', 'pie')),
    ('Does X have the minimum area under the curve?', ('line',)),
    ('Does X have the maximum area under the curve?', ('line',)),
    ('Is X the smoothest?', ('line',)),
    ('Is X the roughest?', ('line',)),
    ('Does X have the lowest value?', ('line',)),
    ('Does X have the highest value?', ('line',)),
    ('Is X less than Y?', ('line',)),
    ('Is X greater than Y?', ('line',)),
    ('Does X intersect Y?', ('line',)),
]

def _load_state(target_model, source_state_dict):
    '''
    Normally `target_model.load_state_dict(source_state_dict)` would suffice,
    but I want to load source dicts that might have been saved from
    a `DataParallel` wrapper around the class of `target_model`.
    '''
    # DataParallel saves its child module into self.module
    class DPWrapper(nn.Module):
        def __init__(self, module):
            super(DPWrapper, self).__init__()
            self.module = module
    try:
        target_model.load_state_dict(source_state_dict)
    # if a DP wrapper then state dict keys are prefixed with 'module.'
    except KeyError:
        wrap = DPWrapper(target_model)
        wrap.load_state_dict(source_state_dict)

def load_model(model_args=None, fname=None, return_args=False, ngpus=0):
    '''
    Create a new RelNet according to provided model arguments.
    Alternatively, loaded model arguments/parameters from a file.

    Arguments:
        model_args: model arguments dict
        fname: name of file to load model arguments/parameters from
        return_args: True to return model arguments
        ngpus: 0 - use cpu
               1 - use a gpu
               >1 - use multiple gpus (nn.DataParallel)
    '''
    from ..models import RelNet
    if model_args == fname:
        raise Exception('To load a model provide either model_args or the '
                        'path of a model checkpoint.')
    if fname is not None:
        print('Loading model from {}'.format(fname))
        mdict = torch.load(fname)
        model_args = mdict['model_args']
    rn = RelNet(model_args)
    # Note: This does NOT maintain the DataParallel device configuration
    # from the source even if the target is DataParallel.
    _load_state(rn, mdict['state_dict'])
    if ngpus:
        rn = rn.cuda()
        if ngpus > 1:
            # devices managed with $CUDA_VISIBLE_DEVICES
            device_ids = list(range(ngpus))
            rn = nn.DataParallel(rn, device_ids=device_ids)
    if return_args:
        return rn, model_args
    else:
        return rn
