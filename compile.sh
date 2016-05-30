#!/usr/bin/env bash
#gcc -O1 -g -Wall -Werror -std=gnu11 -march=native -DEBUG -pthread pagerank.c -o pagerank -lm
#gcc -O1 -g -Wall -Werror -std=gnu11 -march=native -pthread pagerank.c -o pagerank -lm
gcc -O1 -Wall -Werror -std=gnu11 -march=native -pthread pagerank.c -o pagerank -lm
#gcc -O1 -g -pg -Wall -Werror -std=gnu11 -march=native -pthread pagerank.c -o pagerank -lm
#-D__USE_MINGW_ANSI_STDIO=1 -DHAVE_STRUCT_TIMESPEC -std=gnu11 -march=native -lm -pthread pagerank.c -o pagerank
#-fsanitize=address
