#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/flops_line.eps"

set pointintervalbox 3

set logscale x 2
set grid
set autoscale

set title 'ResNet18 FLOPS'
set xlabel 'Batch Size'
set ylabel 'GFLOPS'

file1='data/2080_raw_cifar_flops.txt'
file2='data/Jetson_raw_cifar_flops.txt'
file3='data/cpu_raw_cifar_flops.txt'
plot file1 using 1:5:(sprintf("(%d)", $2)) title "RTX 2080" with linespoints ls 10, \
 file2 using 1:5:(sprintf("(%d)", $2)) title "Jetson Xavier" with linespoints ls 11, \
# file3 using 1:5:(sprintf("(%d)", $2)) title "i7 8700-12 Threads" with linespoints ls 3