srun --partition=Gveval-P1 --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=down --kill-on-bad-exit=1 --quotatype=spot --mail-type=ALL \
 python test.py \