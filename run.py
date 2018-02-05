#!/usr/bin/env python3
from subprocess import Popen
import time
import tempfile
import socket
import argparse

parser = argparse.ArgumentParser(description='Generic slurm run script')
parser.add_argument('-w', '--node', help='name of the machine to run this on')
args = parser.parse_args()

Popen('mkdir -p data/cmds/', shell=True).wait()
Popen('mkdir -p data/logs/', shell=True).wait()

jobid = 0
def runcmd(cmd):
    '''
    Run cmd, a string containing a command, in a bash shell using gpus
    allocated by slurm. Frequently changed slurm settings are configured
    here. 
    '''
    global jobid
    log_fname = 'data/logs/job_{}_{}.log'.format(int(time.time()), jobid)
    jobid += 1
    # write SLURM job id then run the command
    write_slurm_id = True
    if write_slurm_id:
        script_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                             dir='./data/cmds/', prefix='.', suffix='.slurm.sh')
        script_file.write('echo "slurm job id: $SLURM_JOB_ID"\n')
        script_file.write('echo ' + cmd + '\n')
        script_file.write('echo "host: $HOSTNAME"\n')
        script_file.write('echo "cuda: $CUDA_VISIBLE_DEVICES"\n')
        #script_file.write('nvidia-smi -i $CUDA_VISIBLE_DEVICES\n')
        script_file.write(cmd)
        script_file.close()
        # use this to restrict runs to current host
        #hostname = socket.gethostname()
        #cmd = ' -w {} bash '.format(hostname) + script_file.name
        cmd = 'bash ' + script_file.name

    srun_prefix = 'srun --gres gpu:3 '
    if args.node:
        srun_prefix += '-w {} '.format(args.node)

    ############################################################################
    # uncomment the appropriate line to configure how the command is run
    #print(cmd)
    # debug to terminal
    #Popen(srun_prefix + '-p debug --pty ' + cmd, shell=True).wait()
    # debug to log file
    #Popen(srun_prefix + '-p debug -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)
    # debug on current machine without slurm (manually set CUDA_VISIBLE_DEVICES)
    Popen(cmd, shell=True).wait()

    # short queue (no stdout; choose as default)
    #Popen(srun_prefix + '-p short -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)

    # long queue
    #Popen(srun_prefix + '-p long -o {} --open-mode=append '.format(log_fname) + cmd, shell=True)


#######################################
# Config
figureqa_dir = 'data/figureqa/'
figureqa_pre = 'data/figureqa_pre/'

#######################################
# Pre-processing
runcmd('python scripts/prepro_text.py \
            --figqa-dir {figureqa_dir} \
            --output-dir {figureqa_pre}'.format(**locals()))

#######################################
# Train
#runcmd('python train.py \
#        --figqa-dir {figureqa_dir} \
#        --figqa-pre {figureqa_pre} \
#        --model rn \
#        --env-name figqa-rn \
#        --ques-num-layers 1 \
#        --img-net-dim 64 \
#        --rn-g-dim 256 \
#        --ques-rnn-hidden-dim 256 \
#        --batch-size 128 \
#        --shuffle-train 1 \
#        --lr 0.00002 \
#        --lr-decay 1.0'.format(**locals()))

#######################################
# Evaluate
#checkpoint = 'data/checkpoints/<time>/model_ep<epoch>.pt'
#short_name = 'rn'
#for split in ['train1', 'validation1', 'validation2']:
#    runcmd('python evaluate.py \
#            --val-split {split} \
#            --figqa-dir {figureqa_dir} \
#            --figqa-pre {figureqa_pre} \
#            --start-from {checkpoint} \
#            --result-name {short_name} \
#            --batch-size 500'.format(**locals()))

