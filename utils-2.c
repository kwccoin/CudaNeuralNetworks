// see http://stackoverflow.com/questions/14818084/what-is-the-proper-include-for-the-function-sleep-in-c

//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>

#define WARP_SIZE 16
//#define DEBUG false
//#define DEBUG true

// use this and then if there is -DDEBUG it would be set but if not then it is false!

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef DEBUGU
#define DEBUGU false
#endif


#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

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
            printf("u-1-86 %f ", m[i * width + j]);
        }
        printf("\n");
    }
}
