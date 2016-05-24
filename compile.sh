#!/usr/bin/env bash
gcc -O1 -Wall -Werror -D__USE_MINGW_ANSI_STDIO=1 -DHAVE_STRUCT_TIMESPEC -std=gnu11 -march=native -lm -pthread pagerank.c -o pagerank
#-fsanitize=address