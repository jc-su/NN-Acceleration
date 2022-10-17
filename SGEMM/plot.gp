load 'moreland.pal'
set terminal postscript eps enhanced size  3.5in ,2.8in color 
set output "figures/Prefetching.eps"

set pointintervalbox 3
set logscale x 2
set grid
set autoscale
set key Left top left

set title 'SGEMM Optimization'
set xlabel 'Matrix Size [M=N=K]'
set ylabel 'GFLOPS'
# set yrange [0 : 10000]

file0='data/res_0.txt'
file1='data/res_1.txt'
file2='data/res_2.txt'
file3='data/res_3.txt'
file4='data/res_4.txt'
file5='data/res_5.txt'
file6='data/res_6.txt'
file7='data/res_7.txt'
file8='data/res_8.txt'
file9='data/res_9.txt'
file10='data/res_10.txt'
file11='data/res_11.txt'

plot file2 using 1:3:(sprintf("(%d)", $2)) title "Naive" with linespoints ls 1, \
    file4 using 1:3:(sprintf("(%d)", $2)) title "Tiling" with linespoints ls 2, \
    file1 using 1:3:(sprintf("(%d)", $2)) title "Avoid BC" with linespoints ls 4, \
    file7 using 1:3:(sprintf("(%d)", $2)) title "Registers" with linespoints ls 7, \
    file10 using 1:3:(sprintf("(%d)", $2)) title "Prefetching" with linespoints ls 10, \
    # file0 using 1:3:(sprintf("(%d)", $2)) title "cuBLAS" with linespoints ls 12