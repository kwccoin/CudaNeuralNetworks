#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <time.h>

#include "utils.c"
#include "parallel.cu"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

// use this and then if there is -DDEBUG it would be set but if not then it is false!

#ifndef DEBUG
#define DEBUG false
#endif

//#ifdef __APPLE__
//    #include <unistd.h>
//#else _WIN32
//    #include <windows.h>
//#endif

// shall use -DDEBUG and #ifdef DEBUG ... but not
//#define DEBUG true

typedef struct 

    // weights init and bias is an issues

    int n_inputs;
    int n_hidden;
    int n_outputs;

    float *out_input;
    float *out_hidden;
    float *out_output;

    float *changes_input_hidden;
    float *changes_hidden_output;

    float *w_input_hidden;
    float *w_hidden_output;
} NeuralNet;

typedef struct {

    // not sure understand this

    int *result;
    int *data;
} Pattern;

void buildLayer(float *arr, int n, float initial) {
    int i=0;
    while(i < n){
    
        // why change array convention
        // can use arr[] 
        
        *arr = initial;
        arr++;
        i++;
    }
}

float* buildWeightsLayer(int outer_n, int inner_n, float seed) {

     // no bias
     // no allowance of different weights
     //    But if allow defeat the init purpose
     // it should be a 2 dim array
     // weights[inner_layer+1 outer layer] with bias

    int total = outer_n * inner_n;
    float *w = (float *)malloc(sizeof(float) * total);
    for(int i=0; i < total; i++) {
        if (seed == -1) {
          w[i] = ((float)rand()/(float)RAND_MAX);
        } else {
          w[i] = seed;
        }
    }
    return w;
}

NeuralNet buildNeuralNet(int n_inputs, int n_outputs, int n_hidden) {

    float *out_input = (float *)malloc(sizeof(float) * (n_inputs + 1));
    float *out_hidden = (float *)malloc(sizeof(float) * n_hidden);
    float *out_output = (float *)malloc(sizeof(float) * n_outputs);

    buildLayer(out_input, n_inputs + 1, 1.0f);
    buildLayer(out_hidden, n_hidden, 1.0f);
    buildLayer(out_output, n_outputs, 1.0f);

    // Build changes layer
    float *changes_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, 0.0f);
    float *changes_hidden_output = buildWeightsLayer(n_hidden, n_outputs, 0.0f);

    // Build weight matrix
    float *w_input_hidden = buildWeightsLayer(n_inputs + 1, n_hidden, -1.0f);
    float *w_hidden_output = buildWeightsLayer(n_hidden, n_outputs, -1.0f);

    NeuralNet nn;

    nn.n_inputs = n_inputs + 1;
    nn.n_outputs = n_outputs;
    nn.n_hidden = n_hidden;

    nn.out_input = out_input;
    nn.out_hidden = out_hidden;
    nn.out_output = out_output;

    nn.changes_input_hidden = changes_input_hidden;
    nn.changes_hidden_output = changes_hidden_output;

    nn.w_input_hidden = w_input_hidden;
    nn.w_hidden_output = w_hidden_output;

    return nn;
}

float dsigmoid(float y) {
    return 1.0 - pow(y,2.0f);
}

void update_pattern(Pattern pattern, NeuralNet nn) {

    if (DEBUG) {
        printf("\n nn-1-118 ***** LAYER UPDATE *****\n");
    }

    // Write inputs
    int i;
    for(i=0; i < nn.n_inputs -1; i++) {
        nn.out_input[i] = pattern.data[i];
    }

    // Run parallel update
    update_layer(nn.out_input, nn.out_hidden, nn.n_inputs, nn.n_hidden, nn.w_input_hidden);
    update_layer(nn.out_hidden, nn.out_output, nn.n_hidden, nn.n_outputs, nn.w_hidden_output);

    if (DEBUG) {
        printf("\n nn-2-132 ***** END LAYER UPDATE *****\n");
    }
}

float back_propagate_network(Pattern p, NeuralNet n) {

    if (DEBUG) {
        printf("\n nn-3-139 ***** BACK PROPAGATE *****\n");
    }

    int i, j;
    float *output_delta = (float*)malloc(sizeof(float) * n.n_outputs);
    float *hidden_delta = (float*)malloc(sizeof(float) * n.n_hidden);


    // Calculate output delta
    for (i=0; i < n.n_outputs; i++) {
        float error = p.result[i] - n.out_output[i];
        output_delta[i] = dsigmoid(n.out_output[i]) * error;
    }


    // Calculate hidden delta
    for(i=0; i < n.n_hidden; i++) {
        float error = 0.0f;
        for (j=0; j < n.n_outputs; j++) {
            error += output_delta[j] * n.w_hidden_output[i * n.n_outputs + j];
        }
        hidden_delta[i] = dsigmoid(n.out_hidden[i]) * error;
    }

    // Set hidden-output weights
    setWeightsForLayers(n.w_hidden_output, n.changes_hidden_output, output_delta, n.out_hidden, n.n_hidden, n.n_outputs);
    if (DEBUG) {
        printf("\n nn-4-166 Hidden-Output weights\n");
        drawMatrix(n.w_hidden_output, n.n_outputs, n.n_hidden);
        _sleep(1);
    }

    setWeightsForLayers(n.w_input_hidden, n.changes_input_hidden, hidden_delta, n.out_input, n.n_inputs, n.n_hidden);
    if (DEBUG) {
        printf("\n nn-5-173 Input-Hidden weights\n");
        drawMatrix(n.w_input_hidden, n.n_hidden, n.n_inputs);
        _sleep(1);
    }

    // Calculate error
    float error = 0.0f;
    for (i=0; i < n.n_outputs; i++) {
        error = error + 0.5f * pow(p.result[i] - n.out_output[i], 2);
    }
    if (DEBUG) {
        printf("\n nn-6-184 ***** Error for this pattern is: %f *****\n", error);
        _sleep(2);
    }
    return error;
}


void train_network(Pattern *patterns, int n_patterns, int n_iterations, NeuralNet nn) {
  int i, j;
  for (i=0; i < n_iterations; i++) {
    float error = 0;
    for (j=0; j < n_patterns; j++) {
       update_pattern(patterns[j], nn);
       error += back_propagate_network(patterns[j], nn);
    }
    if (i % 10 == 0) {
       printf("nn-7-200 Error is: %-.5f\n", error);
       if (DEBUG) _sleep(2);
    }
  }
}

Pattern makePatternSingleOutput(int *data, int result) {
    Pattern p;
    p.data = data;

    p.result = (int *)malloc(sizeof(int));
    p.result[0] = result;

    return p;
}

int main() {

	printf("nn-8 218 ------------------ starting -------------------------------n");

    srand((unsigned)time(NULL));

    int n_inputs = 2;
    int n_hidden = 4;
    int n_outputs = 1;

    // Build output layer
    NeuralNet nn = buildNeuralNet(n_inputs, n_outputs, n_hidden);

    // Build training samples
    int _p1[] = {0,0};
    Pattern p1 = makePatternSingleOutput(_p1, 1);
    int _p2[] = {0,1};
    Pattern p2 = makePatternSingleOutput(_p2, 0);
    int _p3[] = {1,1};
    Pattern p3 = makePatternSingleOutput(_p3, 1);
    int _p4[] = {1,0};
    Pattern p4 = makePatternSingleOutput(_p4, 0);

    Pattern patterns[] = {p3, p2, p1, p4};

    // Train the network
    train_network(patterns, 4, 1000, nn);

    printf("\n\n nn-9-244 Testing the network\n");
    update_pattern(p2, nn);
    for (int i=0; i < nn.n_outputs; i++) {
        printf(" nn-10-247 Output: %f, expected: %i\n", nn.out_output[i], p2.result[i]);
    }
    cudaDeviceReset();
    return 0;
}
