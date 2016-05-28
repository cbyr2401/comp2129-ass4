#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

#define TOPTIMIAL 50

// vector operations:
double vector_norm(const double* vector_a, const double* vector_b, const ssize_t width, const ssize_t nthreads);
double* vector_sub(const double* vector_a, const double* vector_b, const ssize_t width, const ssize_t nthreads);
void* vector_sub_worker(void* args);
double vector_sumsq(const double* vector, const ssize_t width, const ssize_t nthreads);
void* vector_sumsq_worker(void* argv);

// matrix operations:
double* matrix_init(const double value, ssize_t n, ssize_t nthreads);
void* matrix_init_worker(void* argv);
void matrix_mul(double* result, const double* matrix, const double* vector, const int n, const int nthreads);
void* matrix_mul_worker(void* argv);


// display:
void display(const double* matrix, ssize_t npage);
void display_vector(const double* vector, ssize_t npage);


// matrix struct:
typedef struct {
	double* result;
	const double* matrix;
	const double* vector;
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

	/*  What goes in each (i,j)th cell:
	
						1 / N			if |OUT(j)| = 0  (doesn't link to anything)
			M(i,j)  =   1 / |OUT(j)|	if j links to i  (does link to that page)
						0				otherwise		(doesn't link to that page)
		
		
		As well as this, we need to apply the damping factor (at the same time for speeeeeed)
		
			^M = d*M(i,j) + ((1 - d) / N)
		
		Algorithm: PageRank main
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
	const double add_E = ((1.0 - dampener) / (double)npages);

	// calculate the 1/N value:
	const double div_page = (1.0 / (double)npages);

	#ifdef EBUG
		printf("add_E: %.8lf\n", add_E);
		printf("div_page: %.8lf\n", div_page);
	#endif

	// declare matrix and intialise with given value.
	double* matrix = matrix_init(add_E, npages*npages, nthreads);
	double* p_previous = matrix_init(div_page, npages, nthreads);
	double* p_result;

	// matrix index temp values:
	ssize_t i = 0;
	ssize_t j = 0;

	// node temp variables
	node* current = list;
	node* inlink = NULL;

	// variables for the vector norm of each matrix:
	double norm_result = 0;

	// list of keys:
	char* keys[npages];

	/*
		Algorithm: Building Matrix M and M_hat
		1)  We are at i (from list)
		2)  We can fill in down the matrix (everything in i-th column) if noutlinks = 0
		3)  We can fill in the row using the inlinks of i, so that (i,j)-th entry is 1/|OUT(j)|
		4)  The matrix is initialised as M_hat.
	*/
	while(current != NULL){
		i = current->page->index;

		// add name to list of keys
		keys[i] = current->page->name;

		if(current->page->noutlinks == 0){
			// go down the column putting in the 1/N, adjusted for M_hat
			for(j = 0; j < npages; j++){
				matrix[j * npages + i] += (div_page*dampener);
			}
		}

		inlink = current->page->inlinks;
		while(inlink != NULL){
			// calculate 1 / |OUT(j)| for each inlink page, adjusted for M_hat
			j = inlink->page->index;
			matrix[i * npages + j] += ((1.0 / (double) inlink->page->noutlinks)*dampener);
			inlink = inlink->next;
		}

		// move to next page
		current = current->next;

		#ifdef EBUG
			// display the matrix each iteration (debug)
			printf("matrix building...\n");
			display(matrix, npages);
			printf("\n");
		#endif
	}

	// We now have the matrix M_hat ready to go...let's start the pagerank iterations.
	/*
			Algorithm: Perform iterations (0 to infinity)
		     i) Multiply matrix by P(t)
		    ii) Calculate vector norm for "result" P(t+1) -- cache it for later use.
		   iii) Check for convergence (using cached value and calculated one)
		    iv) Either, break if true or run again if false.
			 v) Store updated cached value for || P(t) ||
			vi) Free the previous P(t) and set P(t+1) TO p(t)
	*/

	#ifdef EBUG
		display_vector(p_previous, npages);
		printf("\n");
	#endif


	while(1){
		p_result = malloc(sizeof(double)*npages);

		matrix_mul(p_result, matrix, p_previous, npages, nthreads);

		#ifdef EBUG
			display_vector(p_result, npages);
			printf("\n");
		#endif

		// calculate the vector norm.
		norm_result = vector_norm(p_result, p_previous, npages, nthreads);

		// check for convergence
		if(norm_result <= EPSILON) break;

		// set up for next iteration...
		free(p_previous);
		p_previous = p_result;
		p_result = NULL;

	}

	#ifdef EBUG
		display_vector(p_result, npages);
		printf("\n");
		printf("\n");
	#endif

	// display results...
	for(i=0; i < npages; i++){
		printf("%s %.8lf\n", keys[i], p_result[i]);
	}

	// free everything...
	free(matrix);
	free(p_result);
	free(p_previous);

}

/**
 *	Calculates the vector norm of the subtraction of two vectors.
 *		Formula: || P(1) - P(0) ||
 */
double vector_norm(const double* vector_a, const double* vector_b, const ssize_t width, const ssize_t nthreads){
	double result = 0.0;

	double* vector = vector_sub(vector_a, vector_b, width, nthreads);

	#ifdef EBUG
		printf("vector norm calc:\n");
		display_vector(vector, width);
		printf("\n");
	#endif

	result = vector_sumsq(vector, width, nthreads);
	result = sqrt(result);
	free(vector);

	return result;
}


/**
 *	Performs vector sum squared on single vector.  Threaded
 *		Formula: sum += vector[i] * vector[i], for all i
 */
double vector_sumsq(const double* vector, const ssize_t width, const ssize_t nthreads){
	double sum = 0.0;
	// initialise arrays for threading
	pthread_t thread_ids[nthreads];
	threadargs args[nthreads];

	// get a function pointer to worker thread:
	void* (*worker)(void*);
	worker = &vector_sumsq_worker;

	// initalise ranges
	int start = 0;
	int end = 0;

	// set arguments for worker
	for(int id=0; id < nthreads; id++){
		end = id == nthreads - 1 ? width : (id + 1) * (width / nthreads);
		args[id] = (threadargs) {
			.result = NULL,
			.vector = vector,
			.start = start,
			.width = width,
			.end = end,
		};
		start = end;
	}

	// launch threads
	for (int i = 0; i < nthreads; i++) pthread_create(thread_ids + i, NULL, worker, args + i );

	// wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);

	// sum up results
	for (size_t i = 0; i < nthreads; i++){
		sum += *(args[i].result);
		free(args[i].result);
	}

	#ifdef EBUG
		printf("sumsq: %.8lf\n", sum);
	#endif

	return sum;

}


/**
 *	Thread Worker for "vector_sumsq"
 */
void* vector_sumsq_worker(void* argv){
	threadargs* data = (threadargs*) argv;

	const int start = data->start;
	const int end = data->end;

	const double* vector = data->vector;
	double* result = malloc(sizeof(double));
	double sum = 0.0;

	for(int i=start; i < end; i++){
		sum += vector[i]*vector[i];
	}
	*result = sum;
	data->result = result;

	return NULL;
}


/**
 *	Initialises a matrix to the given value.
 */
double* matrix_init(const double value, ssize_t n, ssize_t nthreads){
	double* matrix = (double*) malloc((n) * sizeof(double));

	if(n < TOPTIMIAL){
		for(int i=0; i < n; i++){
			matrix[i] = value;
		}
	}else{
		// initialise arrays for threading
		pthread_t thread_ids[nthreads];
		threadargs args[nthreads];

		// get a function pointer to worker thread:
		void* (*worker)(void*);
		worker = &matrix_init_worker;

		// initalise ranges
		int start = 0;
		int end = 0;

		// set arguments for worker
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? n : (id + 1) * (n / nthreads);
			args[id] = (threadargs) {
				.result = matrix,
				.matrix = &value,
				.start = start,
				.width = n,
				.end = end,
			};
			start = end;
		}

		// launch threads
		for (int i = 0; i < nthreads; i++) pthread_create(thread_ids + i, NULL, worker, args + i );

		// wait for threads to finish
		for (size_t i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);
	}

	return matrix;
}


/**
 *	Thread Worker for "matrix_init"
 */
void* matrix_init_worker(void* argv){
	threadargs* data = (threadargs*) argv;

	const int start = data->start;
	const int end = data->end;

	const double value = *(data->matrix);
	double* matrix = data->result;

	for(int i=start; i < end; i++){
		matrix[i] = value;
	}

	return NULL;
}


/**
 *	Performs vector subtraction between two given vectors.  Threaded
 *		Formula: P(1) - P(0)
 */
double* vector_sub(const double* vector_a, const double* vector_b, const ssize_t width, const ssize_t nthreads){
	double* vector = malloc(sizeof(double)*width);

	// initialise arrays for threading
	pthread_t thread_ids[nthreads];
	threadargs args[nthreads];

	// get a function pointer to worker thread:
	void* (*worker)(void*);
	worker = &vector_sub_worker;

	// initalise ranges
	int start = 0;
	int end = 0;

	// set arguments for worker
	for(int id=0; id < nthreads; id++){
		end = id == nthreads - 1 ? width : (id + 1) * (width / nthreads);
		args[id] = (threadargs) {
			.result = vector,
			.matrix = vector_a,
			.vector = vector_b,
			.start = start,
			.width = width,
			.end = end,
		};
		start = end;
	}

	// launch threads
	for (int i = 0; i < nthreads; i++) pthread_create(thread_ids + i, NULL, worker, args + i );

	// wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);

	return vector;
}


/**
 *	Thread Worker for "vector_sub"
 */
void* vector_sub_worker(void* argv){
	threadargs* data = (threadargs*) argv;

	const int start = data->start;
	const int end = data->end;

	const double* vector_a = data->matrix;
	const double* vector_b = data->vector;
	double* vector = data->result;

	for(int i=start; i < end; i++){
		vector[i] = vector_a[i] - vector_b[i];
	}

	return NULL;
}


/**
 *	Matrix Multiply Thread Controller Process
 */
void matrix_mul(double* result, const double* matrix, const double* vector, const int n, const int nthreads){
	// initialise arrays for threading
	pthread_t thread_ids[nthreads];
	threadargs args[nthreads];
	
	// get a function pointer to worker thread:
	void* (*worker)(void*);
	worker = &matrix_mul_worker;
	
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
	for (int i = 0; i < nthreads; i++) pthread_create(thread_ids + i, NULL, worker, args + i );

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
	
	const double* matrix = data->matrix;
	const double* vector = data->vector;
	double* result = data->result;
	
	long double sum = 0.0;
	
	// only use for a matrix * vector ( ^M * P(t) )
	for(int i=start; i < end; i++){
		sum = 0.0;
		for(int j=0; j < width; j++){
			sum += (matrix[i * width + j]*vector[j]);
		}
		result[i] = sum;
	}
	
	return NULL;
}


/**
 * Displays given matrix.
 */
void display(const double* matrix, ssize_t npage) {
	for (ssize_t y = 0; y < npage; y++) {
		for (ssize_t x = 0; x < npage; x++) {
			if (x > 0) printf(" ");
			printf("%.8lf", matrix[y * npage + x]);
		}

		printf("\n");
	}
}


/**
 * Displays given vector.
 */
void display_vector(const double* vector, ssize_t npage) {
	for (ssize_t x = 0; x < npage; x++) {
		printf("%.8lf", vector[x]);
		printf("\n");
	}
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
