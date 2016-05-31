./compile.sh
#./pagerank 2 < tests/sample.in
#time ./pagerank 4 < tests/test06.in | diff - tests/test06.out
#./pagerank 4 < tests/cdog-small.in
#./pagerank 2 < tests/test04.in
valgrind --show-leak-kinds=all --leak-check=full ./pagerank 4 < tests/test06.in
#gprof pagerank
#head tests/test04.out
