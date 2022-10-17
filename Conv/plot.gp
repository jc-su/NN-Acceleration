load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.2in color 
set output "figures/gemm-winograd.eps"

set pointintervalbox 3
set logscale x 2
set grid
set autoscale
set key left top

set title 'Convolution Optimazition'
set xlabel 'Matrix Size'
set ylabel 'Runtime [Milliseconds]'

file1='gemm.txt'
file2='winograd.txt'


plot file1 using 1:3:(sprintf("(%d)", $2)) title "GEMM" with linespoints ls 1, \
    file2 using 1:3:(sprintf("(%d)", $2)) title "Winograd" with linespoints ls 2, \
