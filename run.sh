./compile.sh
#./pagerank 2 < tests/sample.in
time ./pagerank 4 < tests/test12.in | diff - tests/test12.out
#valgrind --show-leak-kinds=all --leak-check=full ./pagerank 2 < tests/sample.in
#head tests/sample.out
