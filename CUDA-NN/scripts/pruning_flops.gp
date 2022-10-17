#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/pruning_flops.eps"

set pointintervalbox 3

# set logscale x 2
set grid
set autoscale
set yrange [ 400 : 1400 ]
set xrange [ 0 : 11 ]

set title 'Pruning - Batchsize: 32'
set xlabel 'Pruning Iteration'
set ylabel 'GFLOPS'

file1='data/l1_pruning_cifar.txt'
file2='data/l2_pruning_cifar.txt'

plot file1 using 1:5:(sprintf("(%d)", $2)) title "L1 Pruning" with linespoints ls 5, \
 file2 using 1:5:(sprintf("(%d)", $2)) title "L2 Pruning" with linespoints ls 6, \