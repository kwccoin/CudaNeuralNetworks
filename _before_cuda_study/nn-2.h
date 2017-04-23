#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

// use this and then if there is -DDEBUG it would be set but if not then it is false!

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef DEBUG2
#define DEBUG2 false
#endif

#ifndef DEBUG2c
#define DEBUG2c false
#endif

#ifndef DEBUGP
#define DEBUGP false
#endif

#ifndef NO_OF_RUN
#define NO_OF_RUN 1000
#endif

#ifndef NO_INPUT_NEURON
#define NO_INPUT_NEURON 2
#endif

#ifndef NO_OUTPUT_NEURON
#define NO_OUTPUT_NEURON 2
#endif

#ifndef NO_HIDDEN_NEURON
#define NO_HIDDEN_NEURON 2
#endif


//#ifdef __APPLE__
//    #include <unistd.h>
//#else _WIN32
//    #include <windows.h>
//#endif

// use this and then if there is -DDEBUG it would be set but if not then it is false!

