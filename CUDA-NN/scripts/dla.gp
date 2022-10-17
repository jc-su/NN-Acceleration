#!/bin/gnuplot

load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/dla.eps"

set pointintervalbox 3

set logscale x 2 
set logscale y 10 
set grid
set autoscale
set yrange [  : 140000 ]

set title 'ResNet18 Jetson Xavier NX with TensorRT'
set xlabel 'Batch Size'
set ylabel 'Throughput [fps]'

file1='data/DLA_CUDA.txt'


plot file1 using 1:2 title "DLA" with linespoints ls 9, \
 file1 using 1:3 title "CUDA" with linespoints ls 10, \