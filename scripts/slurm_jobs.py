#!/usr/bin/env python
import subprocess
import glob
import re

def main():
    '''
    Usage:
        slurm_jobs.py [<id>]
    '''
    import docopt, textwrap
    args = docopt.docopt(textwrap.dedent(main.__doc__))
    if args['<id>']:
        jid = int(args['<id>'])
    else:
        jid = None

    # figure out which jobs to show
    if jid is None:
        output = subprocess.check_output('squeue -hu $USER -o %i', shell=True)
        show_job_ids = list(map(int, output.decode().strip().split('\n')))
    else:
        show_job_ids = [jid]

    # show jobs
    job_logs = []
    job_commands = []
    job_log_ids = []
    for fname in glob.glob('data/logs/*.log'):
        with open(fname, 'r') as f:
            lines = iter(f)
            line = next(lines)
            match = re.match('slurm job id: (\d+)', line)
            if match:
                job_id = int(match.group(1))
                if job_id in show_job_ids:
                    job_logs.append(fname)
                    job_log_ids.append(job_id)
                    job_commands.append(next(lines))

    for jid, log, cmd in zip(job_log_ids, job_logs, job_commands):
        print(jid)
        print(log)
        print(cmd)


if __name__ == '__main__':
    main()

