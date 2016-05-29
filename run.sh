./compile.sh
#./pagerank 2 < tests/sample.in
#time ./pagerank 4 < tests/test12.in | diff - tests/test12.out
./pagerank 4 < tests/test12.in
#./pagerank 2 < tests/test04.in
#valgrind --show-leak-kinds=all --leak-check=full ./pagerank 1 < tests/sample.in
#gprof pagerank
#head tests/test04.out
