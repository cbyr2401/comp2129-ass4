./compile.sh
#./pagerank 2 < tests/sample.in
#./pagerank 1 < tests/sample.in | diff - tests/sample.out
valgrind --show-leak-kinds=all --leak-check=full ./pagerank 2 < tests/sample.in
head tests/sample.out
