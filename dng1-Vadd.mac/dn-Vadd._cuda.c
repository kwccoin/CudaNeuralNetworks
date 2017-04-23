#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

void VectorAdd(int *a, int *b, int *c, int n) 
{

	int i;
	
	for (i = 0; i < n; ++i)
		c[i] = a[i] + b[i];

}

int main(int argc, char *argv[])
{

	int noOfRun;
	if (argc > 1)
		{noOfRun = atoi(argv[1]);
		printf("\nargv[1] in intger=%d\n\n",noOfRun);}
		
	// use SIZE here instead of noofRun

	int *a, *b, *c;
	
	a = (int *)malloc(SIZE*sizeof(int));
	b = (int *)malloc(SIZE*sizeof(int));
	c = (int *)malloc(SIZE*sizeof(int));

	for( int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i+1;
		c[i] = 0;
	}
	
	VectorAdd(a, b, c, SIZE);
	
	for( int i=0; i < 10; ++i)
		printf("%d: a[%d] + b[%d] = %d + %d = c[%d] = %d\n", i, i , i, a[i], b[i], i, c[i]);
	
	free(a);
	free(b);
	free(c);
	
	return 0;
}