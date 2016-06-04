#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>

#include "pagerank.h"

#define TOPTIMIAL 50

// vector operations:
double vector_norm(double* vector, const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads);
double vector_sumsq(const double* vector, const size_t width, const size_t nthreads);
void* vector_sumsq_worker(void* argv);
double* vector_sub(double* vector, const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads);
void* vector_sub_worker(void* argv);

// matrix operations:
void matrix_mul(double* result, const double* matrix, const double* vector, const size_t width, const size_t height, const size_t nthreads);
void* matrix_mul_worker(void* argv);

// reduction operations:
int sortcmp(const void * a, const void * b);
int list_compare(size_t** list, const int row);

#ifdef EBUG
// extra methods for debugging:
void display(const double* matrix, size_t npage);
void display_matrix(const double* matrix, size_t rows, size_t npages);
void display_vector(const double* vector, size_t npage);
#endif

// matrix struct:
typedef struct {
	double* result;
	const double* matrix;
	const double* vector;
	int width;
	int start;
	int end;
} threadargs;

typedef struct {
	double result;
	const double* matrix;
	const double* vector;
	int width;
	int start;
	int end;
} vectorargs;



void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {
	/*
		***********************
		*  PageRank Overview  *
		***********************
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Section 1:  Matrix "M"
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		What goes in each (i,j)th cell:

						1 / N			if |OUT(j)| = 0  (doesn't link to anything)
			M(i,j)  =   1 / |OUT(j)|	if j links to i  (does link to that page)
						0				otherwise		 (doesn't link to that page)
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Section 2: Matrix "M^"
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		As well as this, we need to apply the damping factor (at the same time for speeeeeed)

			^M = d*M(i,j) + ((1 - d) / N)

		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Section 3: "General Algorithm"
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Algorithm: PageRank main
		1) Calculate |OUT| and |IN| sets
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

	// Pre-calculating some values that will be used a number of times...
	const double add_E = ((1.0 - dampener) / (double)npages);	// used for calculating M-hat (i,j)th entries
	const double div_page = (1.0 / (double)npages);				// calculate 1/N
	const double one_on_N = div_page*dampener;					// calculate dampened value for 1/N

	// declare matrix and P(0) and intialise with given value.
	const int nnpages = npages*npages;
	const int nnpages_size = nnpages*sizeof(double);
	const int npages_size = npages*sizeof(double);

	double* matrix = (double*) malloc(nnpages_size);
	double* p_previous = (double*) malloc(npages_size);
	double* p_result = NULL;
	double* p_built = (double*) malloc(npages_size);
	double* vector = (double*) malloc(npages_size);

	// matrix index temp values:
	size_t i = 0;
	size_t j = 0;

	// initialise matrix and P(0)
	for(int i=0; i < npages; i++) p_previous[i] = div_page;
	for(int i=0; i < nnpages; i++)	matrix[i] = add_E;

	// node temp variables
	node* current = list;
	node* inlink = NULL;
	page* cpage = NULL;

	// variables for the vector norm of each matrix:
	double norm_result = 1.0;

	// list of keys:
	char* keys[npages];

	// list of IN() sets for each index.  used for matrix reduction algorithm.
	size_t* in_list[npages];
	size_t inlink_counter = 1;

	// reduction algorithm (on rows only):
	size_t map[npages];  // maps pages --> matrix_row indexes
	double temp_row[npages];
	int row_count = 0;
	int isSame = -1;

	// build the IN() set sub-arrays
	for(int i=0; i < npages; i++) in_list[i] = (size_t*) malloc(sizeof(size_t));

	/*
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Section 4: "Building the Matrix"
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Algorithm: Building Matrix M and M_hat
		1)  We are at i (from list)
		2)  We can fill in down the matrix (everything in i-th column) if noutlinks = 0
		3)  We can fill in the row using the inlinks of i, so that (i,j)-th entry is 1/|OUT(j)|
		4)  The matrix is initialised as M_hat.
	*/

	while(current != NULL){
		// cache the current page, and current index...
		cpage = current->page;
		i = cpage->index;

		// add name to list of keys...
		keys[i] = cpage->name;

		if(cpage->noutlinks == 0){
			// go down the column putting in the 1/N, adjusted for M_hat
			for(int j = 0; j < npages; j++) matrix[j * npages + i] = one_on_N + add_E;
		}

		inlink = cpage->inlinks;

		// backup of current row
		if(row_count > 0){for(int i=0; i < npages; i++) temp_row[i] = matrix[row_count*npages+i];}

		while(inlink != NULL){
			// calculate 1 / |OUT(j)| for each inlink page, adjusted for M_hat
			j = inlink->page->index;

			// add the current "inlink" element to the list IN() set for the i-th element (above)
			in_list[i] = (size_t*) realloc(in_list[i], sizeof(size_t)*(inlink_counter+1));
			in_list[i][inlink_counter++] = j;

			// set the value in the matrix based on: d * ( 1 / |OUT(j)| )
			matrix[row_count * npages + j] = ((1.0 / (double) inlink->page->noutlinks)*dampener) + add_E;

			// move to next inlink
			inlink = inlink->next;
		}

		// sort the list of IN() sets for the i-th element,
		//   remembering to set the first element as the size of the list (because C).
		in_list[i][0] = -1;
		qsort(in_list[i], inlink_counter, sizeof(size_t), sortcmp);
		in_list[i][0] = inlink_counter-1;

		isSame = list_compare(in_list, i);	// Check the in-lists for rows that have the same values

		if(isSame != -1){						// index has the same IN() set as another index.
			map[i] = map[isSame];				// map the index to one that has already been added to the map.
			for(int i=0; i < npages; i++) matrix[row_count*npages+i] = temp_row[i];		// rollback
			in_list[i][0] = 0;
		}else{									// index does not have the same IN() set as another index.
			map[i] = row_count++;				// give the index a new row number for the new smaller matrix
		}

		inlink_counter = 1;

		// move to next page
		current = current->next;
	}

	// set new number of rows
	size_t nrows = row_count;

	// free all the IN() set lists (since we don't need them anymore)
	for(int i=0; i < npages; i++) free(in_list[i]);


	// COMMENT: We now have the matrix M_hat ready to go...let's start the pagerank iterations.
	/*
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Section 5: "PageRank Power Method"
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Algorithm: Power Method / Iterations (0 to infinity)
		 i) Multiply matrix by P(t)
		ii) Calculate vector norm for "result" P(t+1) -- cache it for later use.
	   iii) Check for convergence (using cached value and calculated one)
		iv) Either, break if true or run again if false.
		 v) Store updated cached value for || P(t) ||
		vi) Free the previous P(t) and set P(t+1) TO p(t)
	*/
	// allocate memory for the vector returned by the matrix multiplication...
	p_result = (double*) malloc(sizeof(double)*nrows);

	while(1){
		// multiply the matrix by P(t)
		matrix_mul(p_result, matrix, p_previous, npages, nrows, nthreads);

		// remap the values to a new vector, due to reduced number of rows.
		//   m x 1 vector --> n x 1 vector  where m <= n.
		for(int i = 0; i < npages; i++) p_built[i] = p_result[map[i]];

		// calculate the vector norm.
		norm_result = vector_norm(vector, p_built, p_previous, npages, nthreads);

		// check for convergence
		if(norm_result <= EPSILON) break;

		// set up for next iteration...
		//for(int i = 0; i < npages; i++) p_previous[i] = p_built[i];
		memcpy(p_previous, p_built, npages_size);

	}

	// display results...(mapping to old matrix using the "map")
	for(i=0; i < npages; i++) printf("%s %.8lf\n", keys[i], p_built[i]);

	// free everything...
	free(matrix);
	free(p_result);
	free(p_built);
	free(p_previous);
	free(vector);
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
inline int list_compare(size_t** list, const int row){
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
 *	Calculates the vector norm of the subtraction of two vectors.
 *		Formula: || P(1) - P(0) ||
 */
double vector_norm(double* vector, const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads){
	double result = 0.0;

	vector_sub(vector, vector_a, vector_b, width, nthreads);

	result = vector_sumsq(vector, width, nthreads);
	result = sqrt(result);

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
	vectorargs args[nthreads];

	// get a function pointer to worker thread:
	void* (*worker)(void*);
	worker = &vector_sumsq_worker;

	// initalise ranges
	int start = 0;
	int end = 0;

	// set arguments for worker
	for(int id=0; id < nthreads; id++){
		end = id == nthreads - 1 ? width : (id + 1) * (width / nthreads);
		args[id] = (vectorargs) {
			.result = 0,
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
		sum += args[i].result;
		//free(args[i].result);
	}

	return sum;
}


/**
 *	Thread Worker for "vector_sumsq"
 */
void* vector_sumsq_worker(void* argv){
	vectorargs* data = (vectorargs*) argv;

	const int start = data->start;
	const int end = data->end;

	const double* vector = data->vector;
	double sum = 0.0;

	for(int i=start; i < end; i++){
		sum += vector[i]*vector[i];
	}

	data->result = sum;

	return NULL;
}


/**
 *	Performs vector subtraction between two given vectors.  Threaded
 *		Formula: P(1) - P(0)
 */
double* vector_sub(double* vector, const double* vector_a, const double* vector_b, const size_t width, const size_t nthreads){
	//double* vector = malloc(sizeof(double)*width);

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
	int j;
	int index = 0;
	// only use for a matrix * vector ( ^M * P(t) )
	for(int i=start; i < end; i++){
		sum = 0.0;
		index = i * width;
		for(j=0; j < width - 4; j += 4){
			sum += (matrix[index + j]*vector[j]
					+ matrix[index + (j+1)]*vector[j+1]
					+ matrix[index + (j+2)]*vector[j+2]
					+ matrix[index + (j+3)]*vector[j+3]);
		}
		for(; j < width; j++) sum+= matrix[i * width + j]*vector[j];
		result[i] = sum;
	}

	return NULL;
}



#ifdef EBUG
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
#endif

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