rm data/res*
for i in {0..11..1}
do
	file_name="data/res_${i}.txt"
	./sgemm_benchmarking $i >> ${file_name}
done
