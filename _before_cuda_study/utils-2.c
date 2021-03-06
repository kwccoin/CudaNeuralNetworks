// see http://stackoverflow.com/questions/14818084/what-is-the-proper-include-for-the-function-sleep-in-c

//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
#include <math.h>
#include "nn-2.h"

// should be 2 as cuda from non_cuda one


/* ---------------- [[HELPER FUNCTIONS FOR DEBUGGING]] ---------------- */

void _sleep(int n) {

  //sleep:
  #ifdef _WIN32
  Sleep(n);
  #else
  usleep(n*1000);  /* sleep for 100 milliSeconds */
  #endif
  
    //#ifdef __APPLE__
        //sleep(n);
    //#else _WIN32
        //sleep(n * 1000);
    //#endif
}

void drawMatrix(float *m, int width, int height) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%f ", m[i * width + j]);
        }
        printf("\n");
    }
}

float sigmoid(float z) {
 return 1.0/(1.0+exp(-z));
}

float dsigmoid(float z) {
	float sigmoidz;
	sigmoidz = sigmoid(z);
	return sigmoidz*(1.0f - sigmoidz);
}

float dtanh(float z) {
 return 1.0f - pow(tanh(z), 2.0f);
}