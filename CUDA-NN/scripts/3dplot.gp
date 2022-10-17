#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/3d.eps"

set pointintervalbox 3

# set logscale x 2
set hidden3d
set dgrid3d qnorm 2
set autoscale
set xrange [-0.2 : 11.2]
set yrange [ 400 : 1400 ]
set zrange [ 75 : 100 ]

set title 'Pruning - Batchsize: 32'
set xlabel 'Pruning Iteration'
set ylabel 'GFLOPS'
set zlabel 'Acc. %'

file1='data/3d1.txt'
file2='data/3d2.txt'

splot  file1 using 1:2:3 title"L1 Pruning" with lines ls 5, \
#  file2 using 1:5:6 title "L2 Pruning" with linespoints ls 6, \