import torch

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

def load_model(model_args=None, fname=None):
    from ..models import RelNet
    if model_args == fname:
        raise Exception('To load a model provide either model_args or the '
                        'path of a model checkpoint.')
    if fname is not None:
        print('Loading model from {}'.format(fname))
        mdict = torch.load(fname)
        model_args = mdict['model_args']
    rn = RelNet(model_args)
    if fname is not None:
        rn.load_state_dict(mdict['state_dict'])
    return rn
