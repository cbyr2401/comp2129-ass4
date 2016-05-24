#!/usr/bin/env bash
gcc -O1 -Wall -Werror -DHAVE_STRUCT_TIMESPEC -D_GNU_SOURCE=1 -std=gnu11 -march=native -lm -pthread pagerank.c -o pagerank
#-fsanitize=address