#!/usr/bin/env bash
gcc -O1 -Wall -Werror -std=gnu11 -march=native -DEBUG -lm -pthread pagerank.c -o pagerank
#-D__USE_MINGW_ANSI_STDIO=1 -DHAVE_STRUCT_TIMESPEC -std=gnu11 -march=native -lm -pthread pagerank.c -o pagerank
#-fsanitize=address
