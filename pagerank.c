#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

void* matrix_mul_worker(void* argv);
void matrix_mul_thread(float* result, const float* matrix, const float* vector, const int n, const int nthreads);

// matrix struct:
typedef struct {
	float* result;
	const float* matrix;
	const float* vector;
	int width;
	int start;
	int end;
} threadargs;

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {

	/*
		TODO

		- implement this function
		- implement any other necessary functions
		- implement any other useful data structures
	*/
	
	// for calculating M-hat (i,j)th entries.
	const double add_E = (1.0 - dampener) * (float)npages;
	
	/*  What goes in each (i,j)th cell:
	
						1 / N			if |OUT(j)| = 0  (doesn't link to anything)
			M(i,j)  =   1 / |OUT(j)|	if j links to i  (does link to that page)
						0				otherwise		(doesn't link to that page)
		
		
		As well as this, we need to apply the damping factor (at the same time for speeeeeed)
		
			^M = d*M(i,j) + ((1 - d) / N)
		
		Algorithm
		-------------
		1) Calculate |OUT| and |IN| sets  <<<<<<<<<< HARD
		2) Build Matrix (including dampener)
		3) Initialise P(0)
		4) Create a "result" vector
	   4b) PROGRAMMING:  the initial value of ||P(0) is always the same, that should NEVER be calculated... it is = 0.5
		5) Perform iterations (0 to infinity)
		     i) Multiply matrix by P(t)
		    ii) Calculate vector norm for "result" P(t+1) -- cache it for later use.
		   iii) Check for convergence (using cached value and calculated one)
		    iv) Either, break if true or run again if false.
			 v) Store updated cached value for || P(t) ||
			vi) Free the previous P(t) and set P(t+1) TO p(t)
	    6) Return P(t) final iteration
	*/
	
}


/**
 *	Matrix Multiply Thread Controller Process
 */
void matrix_mul_thread(float* result, const float* matrix, const float* vector, const int n, const int nthreads){
	// initialise arrays for threading
	pthread_t thread_ids[nthreads];
	threadargs args[nthreads];
	
	// get a function pointer to worker thread:
	void* (*g_matrix_worker)(void*);
	g_matrix_worker = &matrix_mul_worker;
	
	int start = 0;
	int end = 0;
	
	// set arguments for worker
	for(int id=0; id < nthreads; id++){
		end = id == nthreads - 1 ? n : (id + 1) * (n / nthreads);
		args[id] = (threadargs) {
			.result = result,
			.matrix = matrix,
			.vector = vector,
			.start = start,
			.width = n,
			.end = end,
		};
		start = end;
	}
	
	// launch threads
	for (int i = 0; i < nthreads; i++) pthread_create(thread_ids + i, NULL, g_matrix_worker, args + i );

	// wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);
	
	return;
}

/**
 *	Matrix Multiply Thread Worker Process
 */
void* matrix_mul_worker(void* argv){
	threadargs* data = (threadargs*) argv;
	
	const int start = data->start;
	const int end = data->end;
	const int width = data->width;
	
	const float* matrix = data->matrix;
	const float* vector = data->vector;
	float* result = data->result;
	
	float sum = 0;
	
	// only use for a matrix * vector ( ^M * P(t) )
	for(int i=start; i < end; i++){
		sum = 0;
		for(int j=0; j < width; j++){
			sum += matrix[i * width + j]*vector[j];
		}
		result[0 * width + i] = sum;
	}
	
	return NULL;
}

/*
######################################
### DO NOT MODIFY BELOW THIS POINT ###
######################################
*/

int main(int argc, char** argv) {

	/*
	######################################################
	### DO NOT MODIFY THE MAIN FUNCTION OR HEADER FILE ###
	######################################################
	*/

	config conf;

	init(&conf, argc, argv);

	node* list = conf.list;
	size_t npages = conf.npages;
	size_t nedges = conf.nedges;
	size_t nthreads = conf.nthreads;
	double dampener = conf.dampener;

	pagerank(list, npages, nedges, nthreads, dampener);

	release(list);

	return 0;
}
