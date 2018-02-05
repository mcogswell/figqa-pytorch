#!/usr/bin/env python
import pickle as pkl
import glob
from pprint import pprint

def main():
    '''
    Print out some results.
    (Put quotes around the first argument if it has * characters.)

    Usage:
        read_results.py <result_file_glob>
    '''
    import docopt, textwrap
    args = docopt.docopt(textwrap.dedent(main.__doc__))

    fnames = glob.glob(args['<result_file_glob>'])
    print(fnames)
    for fname in fnames:
        with open(fname, 'rb') as f:
            try:
                dat = pkl.load(f)
                print(fname)
                print(dat['args'].checkpoint_dir)
                pprint(dat['acc'])
            except pkl.PickleError:
                print('Could not read {}. Is it a pickle file?'.format(fname))

if __name__ == '__main__':
    main()
