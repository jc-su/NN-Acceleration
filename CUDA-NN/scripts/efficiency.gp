#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/efficiency.eps"

set pointintervalbox 3

set logscale x 2
set grid
set autoscale
# set xrange [0.9:64.1]
set yrange [ 0 : 12 ]

set title 'Perf/W'
set xlabel 'Batch Size'
set ylabel 'GFLOPS/W'

file1='data/2080_Efficiency.txt'
file2='data/Jetson_Efficiency.txt'

plot file1 using 1:2 title "RTX 2080" with linespoints ls 7, \
 file2 using 1:2 title "Jetson" with linespoints ls 8, \