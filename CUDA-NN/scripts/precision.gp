#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/precision.eps"

set pointintervalbox 3

set logscale x 2
# set logscale y 10 
set grid
set autoscale
# set xrange [0.94 : 42]
# set yrange [ : 25000 ]
set key Left top left
set title 'Resnet18 Quantization'
set xlabel 'Batch Size'
set ylabel 'Throughput [fps]'

file1='data/precision.txt'

plot file1 using 1:3:(sprintf("(%d)", $2)) title "FP32" with linespoints ls 7, \
    file1 using 1:5:(sprintf("(%d)", $2)) title "FP16" with linespoints ls 8, \
    file1 using 1:7:(sprintf("(%d)", $2)) title "INT8" with linespoints ls 9, \