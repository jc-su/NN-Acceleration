#!/bin/gnuplot
load 'moreland.pal'
set terminal pngcairo enhanced size  1024,768
set output "data/fps_line.png"

set pointintervalbox 3
set logscale x 2
set grid
set autoscale

set title 'ResNet18 FPS'
set xlabel 'Batch Size'
set ylabel 'FPS'

file1='2080-raw-cifar.txt'
file2='jetson_data.txt'
file3='cpu_data.txt'
plot file1 using 1:3:(sprintf("(%d)", $2)) title "Tesla V100-16GB" with linespoints ls 1, \
 file2 using 1:3:(sprintf("(%d)", $2)) title "Jetson Xavier" with linespoints ls 2, \
 file3 using 1:3:(sprintf("(%d)", $2)) title "i7 8700-12 Threads" with linespoints ls 3