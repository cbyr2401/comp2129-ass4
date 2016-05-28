./compile.sh
#./pagerank 2 < tests/sample.in
#time ./pagerank 4 < tests/test12.in | diff - tests/test12.out
./pagerank 2 < tests/test2.in
#valgrind --show-leak-kinds=all --leak-check=full ./pagerank 4 < tests/sample.in
head tests/test2.out
