file="data/2080_raw_cifar_energy.txt"
echo "Power_Min[W], Power_Max[W], Power_Average[W]" >>$file
for b in 1 2 3 4 6 9 12 17 24 33 46 64; do
    ./bin/energy --input ./images/10.ppm --weights_dir ./python/l2_round0/ --batch_size $b --iters 100 >>$file
    # nvprof --metrics flop_count_dp --metrics sysmem_read_bytes --metrics sysmem_write_bytes ./bin/main --input ./images/cat.ppm --weights_dir ./python/weights/ --batch_size $b --iters 100 >>data.txt
done