./compile.sh
#/pagerank 2 < tests/sample.in | diff - tests/sample.out
#time ./pagerank 4 < tests/cdog-medium.in | diff - tests/cdog-medium.out
time ./pagerank 4 < tests/test12.in | diff - tests/test12.out
#time ./pagerank 2 < tests/test06.in | diff - tests/test06.out
#./pagerank 4 < tests/cdog-medium.in
#./pagerank 2 < tests/test04.in
#valgrind --show-leak-kinds=all --leak-check=full ./pagerank 4 < tests/test12.in | diff - tests/test12.out
#gprof pagerank
#head tests/sample.out
