#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/pruning_acc.eps"

set pointintervalbox 3

# set logscale x 2
set grid
set autoscale
set yrange [ 60 : 100 ]
set xrange [ 0 : 11 ]

set title 'Pruning - Batchsize: 32'
set xlabel 'Pruning Iteration'
set ylabel 'Acc. [%]'

file1='data/3d1.txt'
file2='data/3d2.txt'

plot file1 using 1:3 title "L1 Pruning" with linespoints ls 5, \
 file2 using 1:3 title "L2 Pruning" with linespoints ls 6, \