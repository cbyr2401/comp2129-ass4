#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

#define TOPTIMIAL 50

// vector operations:
double vector_norm(const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads);
double vector_sumsq(const double* vector, const size_t width, const size_t nthreads);
void* vector_sumsq_worker(void* argv);
double* vector_sub(const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads);
void* vector_sub_worker(void* argv);


// matrix operations:
double* matrix_init(const double value, size_t n, size_t nthreads);
void* matrix_init_worker(void* argv);
void matrix_mul(double* result, const double* matrix, const double* vector, const size_t width, const size_t height, const size_t nthreads);
void* matrix_mul_worker(void* argv);


// reduction operations:
double* build_vector(const double* vector, const size_t* map, const size_t npages);
int sortcmp(const void * a, const void * b);
int list_compare(size_t** list, const int row);
double* matrix_reduce(double* matrix, size_t* map, size_t** in_list, size_t* nrows, const size_t npages);
double* remrow_matrix(size_t* nrows, const double* matrix, const size_t* del, const size_t width, const int ndel);

// extra methods for debugging:
void display(const double* matrix, size_t npage);
void display_matrix(const double* matrix, size_t rows, size_t npages);
void display_vector(const double* vector, size_t npage);


// function pointers to threading operations:




// matrix struct:
typedef struct {
	double* result;
	const double* matrix;
	const double* vector;
	int width;
	int start;
	int end;
} threadargs;

// qsort and other
typedef struct{
	int* list;
	int size;
} array;


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
	const double one_on_N = div_page*dampener;

	// declare matrix and intialise with given value.
	double* matrix = matrix_init(add_E, npages*npages, nthreads);
	double* p_previous = matrix_init(div_page, npages, nthreads);
	double* p_result = NULL;
	double* p_built = NULL;

	// matrix index temp values:
	size_t i = 0;
	size_t j = 0;

	// node temp variables
	node* current = list;
	node* inlink = NULL;
	page* cpage = NULL;

	// variables for the vector norm of each matrix:
	double norm_result = 0;

	// list of keys:
	char* keys[npages];
	size_t* in_list[npages];
	size_t inlink_counter = 1;

	/*
		Algorithm: Building Matrix M and M_hat
		1)  We are at i (from list)
		2)  We can fill in down the matrix (everything in i-th column) if noutlinks = 0
		3)  We can fill in the row using the inlinks of i, so that (i,j)-th entry is 1/|OUT(j)|
		4)  The matrix is initialised as M_hat.
	*/

	for(int i=0; i < npages; i++){
		in_list[i] = (size_t*) malloc(sizeof(size_t));
	}

	while(current != NULL){
		cpage = current->page;
		i = cpage->index;

		// add name to list of keys
		keys[i] = cpage->name;

		if(cpage->noutlinks == 0){
			// go down the column putting in the 1/N, adjusted for M_hat
			for(j = 0; j < npages; j++){
				matrix[j * npages + i] += one_on_N;
			}
		}

		inlink = cpage->inlinks;

		while(inlink != NULL){
			// calculate 1 / |OUT(j)| for each inlink page, adjusted for M_hat
			j = inlink->page->index;

			in_list[i] = (size_t*) realloc(in_list[i], sizeof(size_t)*(inlink_counter+1));
			in_list[i][inlink_counter++] = j;

			matrix[i * npages + j] += ((1.0 / (double) inlink->page->noutlinks)*dampener);
			inlink = inlink->next;
		}

		in_list[i][0] = -1;
		qsort(in_list[i], inlink_counter, sizeof(size_t), sortcmp);
		in_list[i][0] = inlink_counter-1;
		inlink_counter = 1;

		// move to next page
		current = current->next;
	}

	// reduction algorithm:
	size_t* map = (size_t*) malloc(sizeof(size_t)*npages);  // maps pages --> matrix_row indexes
	size_t nrows = npages;

	matrix = matrix_reduce(matrix, map, in_list, &nrows, npages);

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


	//int iterations = 0;
	while(1){
		p_result = (double*) malloc(sizeof(double)*nrows);

		matrix_mul(p_result, matrix, p_previous, npages, nrows, nthreads);

		p_built = build_vector(p_result, map, npages);

		// calculate the vector norm.
		norm_result = vector_norm(p_built, p_previous, npages, nthreads);

		// check for convergence
		if(norm_result <= EPSILON) break;

		// set up for next iteration...
		free(p_result);
		free(p_previous);
		p_previous = p_built;
		p_result = NULL;
		//printf("iterations: %u\n", iterations++);
		//sleep(1);

		//if(iterations > 5) exit(0);

	}

	// display results...
	for(i=0; i < npages; i++){
		printf("%s %.8lf\n", keys[i], p_built[i]); //p_result[map[i]]);
	}

	// free everything...
	free(matrix);
	free(p_result);
	free(p_built);
	free(p_previous);
	free(map);
}


/**
 *	Build a larger vector for the next calcuations.
 */
double* build_vector(const double* vector, const size_t* map, const size_t npages){
	double* result = (double*) malloc(sizeof(double)*npages);

	for(int i = 0; i < npages; i++){
		result[i] = vector[map[i]];
	}

	return result;
}

/**
 *	Description:	Compare function for qsort
 *	Return:			difference between two ints
 * 	Source:			http://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm
 */
int sortcmp(const void * a, const void * b){
	return ((int)( *(size_t*)a - *(size_t*)b ));
}


/**
 * 	Compares lists and returns the row_id of the FIRST list that is the same or -1.
 *		We only want to go upto the ones that we have seen.
 */
int list_compare(size_t** list, const int row){
	int i;
	int flag = 0;
	for(i=0; i < row; i++){
		// check sizes:
		if(list[row][0] == list[i][0]){
			// same size, continue:
			for(int j=1; j < list[row][0]+1; j++){
				if(list[row][j] != list[i][j]) {flag= 0; break;}
				else flag = 1;
			}
			if(flag) return i;
		}
	}
	return -1;
}


/**
 * 	Reduce the matrix size
 */
double* matrix_reduce(double* matrix, size_t* map, size_t** in_list, size_t* nrows, const size_t npages){
	size_t nrow_del = 0;
	int isSame = -1;
	size_t row_count = 0;

	//size_t* delete_rows = (size_t*) malloc(sizeof(size_t)*(npages-1));  // rows to delete in matrix.
	size_t delete_rows[npages];

	for(int row_id = 0; row_id < npages; row_id++){
		// Check the in-lists for rows that have the same values
		isSame = list_compare(in_list, row_id);

		// build the map of the rows.  This is very important
		if(isSame != -1){
			delete_rows[nrow_del++] = row_id;
			map[row_id] = map[isSame];
		}else{
			map[row_id] = row_count++;
		}
	}

	double* result = NULL;

	if(nrow_del == 0){
		// this was a huge waste of time... set result to what we received.
		result = matrix;
		*nrows = npages;
	}else{
		// build a new matrix with a reduced number of rows.  Delete the rows in delete_rows[].
		//  Return new matrix and set new number of rows (inside method).
		result = remrow_matrix(nrows, matrix, delete_rows, npages, nrow_del);
		// free the old matrix.
		free(matrix);
	}

	for(int i=0; i < npages; i++){
		free(in_list[i]);
	}

	// return result matrix
	return result;
}


/**
 *	Function to build a new matrix, removing the rows that are not wanted (given in an array).
 */
double* remrow_matrix(size_t* nrows, const double* matrix, const size_t* del, const size_t width, const int ndel){
	// set the new number of rows, which is the old no. rows mnius the no. rows deleted.
	size_t num_rows = width - ndel;

	// keep track of where we are in the del[]
	int next = 0;

	// create a new matrix that is smaller than the given matrix:
	double* result = (double*) malloc(sizeof(double)*(width*num_rows));

	for(int row = 0; row < num_rows; row++){
		if( next < ndel && (row+next) == del[next]){
			// when we hit the row that we want to remove, increment the offset to skip that row.
			next++;
			row--;
		}else{
			// add column to matrix
			for(int col = 0; col < width; col++){
				result[(row) * width + (col)] = matrix[(row+next) * (width) + (col)];
			}
		}

	}

	// set the reduced number of rows..
	*nrows = num_rows;

	// return the new matrix:
	return result;
}


/**
 *	Calculates the vector norm of the subtraction of two vectors.
 *		Formula: || P(1) - P(0) ||
 */
double vector_norm(const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads){
	double result = 0.0;

	double* vector = vector_sub(vector_a, vector_b, width, nthreads);

	result = vector_sumsq(vector, width, nthreads);
	result = sqrt(result);
	free(vector);

	return result;
}


/**
 *	Performs vector sum squared on single vector.  Threaded
 *		Formula: sum += vector[i] * vector[i], for all i
 */
double vector_sumsq(const double* vector, const size_t width, const size_t nthreads){
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
	for (int i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);

	// sum up results
	for (int i = 0; i < nthreads; i++){
		sum += *(args[i].result);
		free(args[i].result);
	}

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
double* matrix_init(const double value, size_t n, size_t nthreads){
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
		for (int i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);
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
double* vector_sub(const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads){
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
	for (int i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);

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
void matrix_mul(double* result, const double* matrix, const double* vector, const size_t width, const size_t height, const size_t nthreads){
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
		end = id == nthreads - 1 ? height : (id + 1) * (height / nthreads);
		args[id] = (threadargs) {
			.result = result,
			.matrix = matrix,
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
	for (int i = 0; i < nthreads; i++) pthread_join(thread_ids[i], NULL);

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

	double sum = 0.0;

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



//#ifdef EBUG
/**
 * Displays given matrix.
 */
void display(const double* matrix, size_t npage) {
	for (ssize_t y = 0; y < npage; y++) {
		for (ssize_t x = 0; x < npage; x++) {
			if (x > 0) printf(" ");
			printf("%.8lf", matrix[y * npage + x]);
		}

		printf("\n");
	}
	printf("\n");
}


/**
 * Displays given matrix.
 */
void display_matrix(const double* matrix, size_t rows, size_t npages) {
	for (ssize_t y = 0; y < rows; y++) {
		for (ssize_t x = 0; x < npages; x++) {
			if (x > 0) printf(" ");
			printf("%.8lf", matrix[y * npages + x]);
		}

		printf("\n");
	}
	printf("\n");
}

/**
 * Displays given vector.
 */
void display_vector(const double* vector, size_t npage) {
	for (ssize_t x = 0; x < npage; x++) {
		printf("%.24lf", vector[x]);
		printf("\n");
	}
	printf("\n");
}
//#endif

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