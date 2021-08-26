../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device igPU --type 0 --report_path /root/dwarf_bench/reports/nested_device_test.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 25600 38400 51200 64000 76800 89600 102400 115200 128000 --device igPU --type 0 --report_path /root/dwarf_bench/reports/nested_device.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 25600 38400 51200 64000 76800 89600 102400 115200 128000 --device cpu --type 0 --report_path /root/dwarf_bench/reports/nested_device.csv
#
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device cpu --type 1 --threads_count 32 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device cpu --type 1 --threads_count 64 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device cpu --type 1 --threads_count 96 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device cpu --type 1 --threads_count 128 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device cpu --type 1 --threads_count 160 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
#
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device igPU --type 1 --threads_count 32 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device igPU --type 1 --threads_count 64 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device igPU --type 1 --threads_count 96 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device igPU --type 1 --threads_count 128 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 --device igPU --type 1 --threads_count 160 --report_path /root/dwarf_bench/reports/nested_type_threads.csv
#
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device cpu --type 1 --threads_count 12800 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device cpu --type 1 --threads_count 25600 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device cpu --type 1 --threads_count 38400 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device cpu --type 1 --threads_count 51200 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#
##
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device igPU --type 1 --threads_count 12800 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device igPU --type 1 --threads_count 25600 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device igPU --type 1 --threads_count 38400 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv
#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 51200 --device igPU --type 1 --threads_count 51200 --report_path /root/dwarf_bench/reports/nested_type_threads_big_step_vs.csv


#../build/dwarf_bench --dwarf NestedLoopJoin --iterations 9 --input_size 12800 25600 38400 51200 64000 76800 89600 102400 115200 128000 --device igpu --type 0 --report_path /root/dwarf_bench/reports/nested_vs_slab.csv
#../build/dwarf_bench --dwarf SlabJoin --iterations 9 --input_size 12800 25600 38400 51200 64000 76800 89600 102400 115200 128000 --device igpu --type 0 --report_path /root/dwarf_bench/reports/nested_vs_slab.csv
