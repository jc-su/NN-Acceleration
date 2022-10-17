START=$((32))
STOP=$((10280))
SAMPLES=64

y=$(echo "e(l($STOP/$START)/$SAMPLES)" | bc -l)

for i in $(seq 0 $SAMPLES); do
    size=$(echo "$START*$y^$i" | bc -l | xargs printf "%f" | cut -d '.' -f 1)
    echo $size
    # python gen_img.py $size $size 1
    ./build/conv_benchmark ./images/$size.jpg >> wino2.txt
done