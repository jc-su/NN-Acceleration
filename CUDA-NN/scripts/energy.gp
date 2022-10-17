#!/bin/gnuplot
load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/energy.eps"

set pointintervalbox 3
set style increment default
set style data lines
set logscale x 2
set grid
set autoscale
set ytics norangelimit logscale 

set xrange [ * : * ] noreverse writeback
set x2range [ * : * ] noreverse writeback

set yrange [ * : * ] noreverse writeback
set y2range [ * : * ] noreverse writeback
set zrange [ * : * ] noreverse writeback
set cbrange [ * : * ] noreverse writeback
set rrange [ * : * ] noreverse writeback
set yrange [ 0 : 110 ]

set title 'Energy consumption'
set xlabel 'Batch Size'
set ylabel 'Power [W]'

file1='data/2080_raw_cifar_energy.txt'
file2='data/Jetson_raw_cifar_energy.txt'

plot file1 using 1:4:2:3 title "RTX 2080" w yerrorlines ls 6, \
 file2 using 1:4:2:3 title "Jetson Xavier" w yerrorlines ls 9
