TODO before release
---
1. visualize some examples in visdom, possibly including this in the logging step
2. (done) check code TODOs
3. (done) lookup checklist for releasing github code (remove history?)
4. (done) lint code
5. (done) update paper numbers to use val numbers from openreview
6. (done) test ngpus argument
7. (done) fill in the running the code section of README
8. replicate results in a couple more runs of the model
9. (done) configure run.py so it does not use slurm by default
10. (done) support with and without visdom? NO
11. (done) note about which pytorch version I'm using
12. (done) fix saving/loading bug using DataParallel

TODO
---
1. Checkpoints save optimizer state, but continuing training
from such a checkpoint is not yet supported.
2. submit test1 and test2 results to be evaluated

