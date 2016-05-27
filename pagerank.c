#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

void* matrix_mul_worker(void* argv);
void matrix_mul_thread(float* result, const float* matrix, const float* vector, const int n, const int nthreads);
float* matrix_init(const float value, ssize_t n);

// matrix struct:
typedef struct {
	float* result;
	const float* matrix;
	const float* vector;
	int width;
	int start;
	int end;
} threadargs;

/**
 * Displays given matrix.
 */
void display(const float* matrix, ssize_t npage) {
	for (ssize_t y = 0; y < npage; y++) {
		for (ssize_t x = 0; x < npage; x++) {
			if (x > 0) printf(" ");
			printf("%.8lf", matrix[y * npage + x]);
		}

		printf("\n");
	}
}

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {

	/*
		TODO

		- implement this function
		- implement any other necessary functions
		- implement any other useful data structures
	*/
	
	/*
	node* current = list;
	node* inlink = NULL;


	ssize_t current_index;
	while(current != NULL){
		printf("****** PAGE DETAILS ******\n");
		printf("name: %s\n", current->page->name);
		printf("index: %zu\n", current->page->index);
		printf("noutlinks: %zu\n", current->page->noutlinks);

		current_index = current->page->index;

		printf("~~~~~~~~ LINKED PAGES ~~~~~~~\n");
		inlink = current->page->inlinks;
		if (inlink != NULL){
			size_t o = malloc(sizeof(ssize_t)*1);
			o[0] = 0;
			int index = 0;
			while(inlink != NULL){
				printf("name: %zu\n", inlink->page->index);
				o[0]++;
				o = realloc(sizeof(ssize_t)*o[0]);
				out[inlink->page->index]
				inlink = inlink->next;

			}
		}
		current = current->next;


	}
	*/



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




	// for calculating M-hat (i,j)th entries.
	const double add_E = ((1.0 - dampener) / (float)npages);

	// calculate the 1/N value:
	const float div_page = (1.0 / (float)npages);

	// declare matrix and intialise with given value.
	float* matrix = matrix_init((float)add_E, npages, npages);
	float* p_previous = matrix_init(div_page, npages, 1);
	float* p_result = NULL;

	// matrix index temp values:
	ssize_t i = 0;
	ssize_t j = 0;

	// node temp variables
	node* current = list;
	node* inlink = NULL;

	// variables for the vector norm of each matrix:
	double norm_previous = 0;  // since the start matrix always has the same vector norm...
	double norm_result = 0;

	/*
		Algorithm: Building Matrix M and M_hat
		1)  We are at i (from list)
		2)  We can fill in down the matrix (everything in i-th column) if noutlinks = 0
		3)  We can fill in the row using the inlinks of i, so that (i,j)-th entry is 1/|OUT(j)|
		4)  The matrix is initialised as M_hat.
	*/
	while(current != NULL){
		i = current->page->index;
		if(current->page->noutlinks == 0){
			// go down the column putting in the 1/N, adjusted for M_hat
			for(j = 0; j < npages; j++){
				matrix[j * npages + i] = (div_page*dampener)+add_E;
			}
		}

		inlink = current->page->inlinks;
		while(inlink != NULL){
			// calculate 1 / |OUT(j)| for each inlink page, adjusted for M_hat
			j = inlink->page->index;
			matrix[i * npages + j] = ((1.0 / (float) inlink->page->noutlinks)*dampener)+add_E;
			inlink = inlink->next;
		}

		// move to next page
		current = current->next;

		#ifdef EBUG
		// display the matrix each iteration (debug)
		display(matrix, npages);
		printf("\n");
		#endif
	}

	// We now have the matrix M_hat ready to go...let's start the pagerank iterations.
	while(1){
		matrix_mul_thread(p_result, matrix, p_previous, npage, nthreads);


	}


}

float* matrix_init(const float value, ssize_t n, ssize_t n2){
	float* matrix = (float*) malloc((n*n) * sizeof(float));

	for(int i=0; i < n*n2; i++){
		matrix[i] = value;
	}

	return matrix;
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
