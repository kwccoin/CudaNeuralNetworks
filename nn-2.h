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


//#ifdef __APPLE__
//    #include <unistd.h>
//#else _WIN32
//    #include <windows.h>
//#endif

// use this and then if there is -DDEBUG it would be set but if not then it is false!

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef DEBUGP
#define DEBUGP false
#endif
