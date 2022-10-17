#!/bin/bash -l
# 1 1 2 2 4 5 8 11 16 22 32 45 64
# batchsize_set=(1 2 4 8 16 32 64)
file="data/2080_raw_cifar_flops.txt"
echo "Bath Size Time FPS BandWidth[GB/S] GFLOPS[GB/S]" >>$file
for b in 1 2 3 4 6 9 12 17 24 33 46 64; do
    ./bin/flops --input ./images/10.ppm --weights_dir ./python/l2_round0/ --batch_size $b --iters 100 >>$file
    # nvprof --metrics flop_count_dp --metrics sysmem_read_bytes --metrics sysmem_write_bytes ./bin/main --input ./images/cat.ppm --weights_dir ./python/weights/ --batch_size $b --iters 100 >>data.txt
done
