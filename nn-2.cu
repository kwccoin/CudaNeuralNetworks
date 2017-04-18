#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <time.h>

#include "nn-2.h"
#include "nn-2_cuda.cu"

// note cannot include one more time utils-2.c

typedef struct {
 
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
    int *result;
    int *data;
} Pattern;

void buildLayer(float *arr, int n, float initial) {
    
    // why this a layer
    // we need layers per neutron layer ?
    
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

    // ok for simple to assume only 1 "layer" of hidden ... need concept extension though
    
    // per each pattern p as input_feeder[p]
    
    // input
    // input2hidden  - fwd: weights and bias
    // hidden 
    // hidden2hidden - fwd: weights and bias
    //.              - bwd: delta (or hidden)
    // hidden
    // hidden2output - fwd: weights and bias
    //.              - bwd: delta (or hidden)
    // output
    //.              - error calc (or in pattern)
    
    // here it use the idea of out_ but need bac
    //.    And also input just has out no bwd
    
    // per each pattern p as expected_output[p]
    
    // batch
    // regularisation
    // era
    // delta 
    // ...
    
    // absolute minimum model is 2i-2h-2h-2o and patterns.  

    float *out_input = (float *)malloc(sizeof(float) * (n_inputs + 1)); // need 1 extra ? got bias
    
    float *out_hidden = (float *)malloc(sizeof(float) * n_hidden); // no 1 extra ? no bias
    
    float *out_output = (float *)malloc(sizeof(float) * n_outputs);

    buildLayer(out_input, n_inputs + 1, 1.0f);  // why plus 1 here ??
    
    buildLayer(out_hidden, n_hidden, 1.0f);
    buildLayer(out_output, n_outputs, 1.0f);

    // Build changes layer ? not sure what is this
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

void print_nn(NeuralNet nn){
	printf("\n--nn start seems input +/-1 is for bias but only for input strangely --\n");
	
	int i; 
	
	for(i=0; i < (nn.n_inputs+1); i++)                {printf(" nn.out_input[%d]: %f\n",             i, nn.out_input[i]);};
	for(i=0; i < (nn.n_hidden); i++)                  {printf(" nn.out_hidden[%d]: %f\n",            i, nn.out_hidden[i]);};
	for(i=0; i < (nn.n_outputs); i++)                 {printf(" nn.out_output[%d]: %f\n",            i, nn.out_output[i]);};
	printf("\n");

	for(i=0; i < ((nn.n_inputs+1)*nn.n_hidden); i++)  {printf(" nn.changes_input_hidden[%d]: %f\n",  i, nn.changes_input_hidden[i]);};
	for(i=0; i < ((nn.n_hidden)  *nn.n_outputs); i++) {printf(" nn.changes_hidden_output[%d]: %f\n", i, nn.changes_hidden_output[i]);};
	printf("\n");

	for(i=0; i < ((nn.n_inputs+1)*nn.n_hidden); i++)  {printf(" nn.w_input_hidden[%d]: %f\n",        i, nn.w_input_hidden[i]);};
	for(i=0; i < ((nn.n_hidden)  *nn.n_outputs); i++) {printf(" nn.w_hidden_output[%d]: %f\n",       i, nn.w_hidden_output[i]);};
    printf("\n");
		
	printf("\n--nn end   --\n");
	
}

void update_pattern(Pattern pattern, NeuralNet nn) {

    if (DEBUG | DEBUG2c) {
        printf("\n DEBUG2-a ***** LAYER UPDATE *****\n");
        print_nn(nn);
    }

    // Write inputs // mixing all 3 togethers
    int i;
    for(i=0; i < (nn.n_inputs -1); i++) {        // -1 here ... why??
        nn.out_input[i] = pattern.data[i];     // why pattern.data[i] here ??? here it will store of these data in out_input[i]
    }

    // Run parallel update and amend to use cuda 
    update_layer_CUDA(nn.out_input, nn.out_hidden, nn.n_inputs, nn.n_hidden, nn.w_input_hidden);
    
    update_layer_CUDA(nn.out_hidden, nn.out_output, nn.n_hidden, nn.n_outputs, nn.w_hidden_output);

    if (DEBUG | DEBUG2) {
        printf("\n DEBUG2-b ***** END LAYER UPDATE *****\n");
    }
}

float back_propagate_network(Pattern p, NeuralNet n) {

    // no parallel? No cuda?? Why not all in cuda once built???

    if (DEBUG | DEBUG2c) {
        printf("\n DEBUG2-c ***** BACK PROPAGATE *****\n");
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
    setWeightsForLayers_CUDA(n.w_hidden_output, n.changes_hidden_output, output_delta, n.out_hidden, n.n_hidden, n.n_outputs);
    if (DEBUG | DEBUG2c) {
        printf("\n DEBUG2-d Hidden-Output weights\n");
        drawMatrix(n.w_hidden_output, n.n_outputs, n.n_hidden);
        _sleep(1);  // why need to sleep ?
    }

    setWeightsForLayers_CUDA(n.w_input_hidden, n.changes_input_hidden, hidden_delta, n.out_input, n.n_inputs, n.n_hidden);
    if (DEBUG | DEBUG2c) {
        printf("\n DEBUG2-e Input-Hidden weights\n");
        drawMatrix(n.w_input_hidden, n.n_hidden, n.n_inputs);
        _sleep(1);  // why need to sleep ?
    }

    // Calculate error
    float error = 0.0f;
    for (i=0; i < n.n_outputs; i++) {
        error = error + 0.5f * pow(p.result[i] - n.out_output[i], 2);
    }
    if (DEBUG | DEBUG2c) {
        printf("\n DEBUG2-f ***** Error for this pattern is: %f *****\n", error);
        _sleep(2); // why need to sleep ?
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
    if (i % 10 == 0 | i < 10) {
       printf("nn-2-235 Error for iter %d is: %-.5f\n", i, error);
       if (DEBUG | DEBUG2) _sleep(2);  // why need sleep ???
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

	printf("nn-2 253 ------------------ main() starting -------------------------------n");

    srand((unsigned)time(NULL));

    int n_inputs  = NO_INPUT_NEURON;   //2;  // shall use configuration ... ???
    int n_outputs = NO_OUTPUT_NEURON;  //1;
	int n_hidden  = NO_HIDDEN_NEURON;  //4 -> 2;
	
	// assume 2 input neuron, 4 hidden neuron and 1 output neuron with bais
	
	// 00b -3x5-> xxxxb -5x1-> 1
	// 01b -3x5-> xxxxb -5x1-> 0
	// 10b -3x5-> xxxxb -5x1-> 1
	// 11b -3x5-> xxxxb -5x1-> 0
    
    // Build output layer
    NeuralNet nn = buildNeuralNet(n_inputs, n_outputs, n_hidden); 

    // Build training samples - real life shall use file ... 
    int _p1[] = {0,0};
    Pattern p1 = makePatternSingleOutput(_p1, 1); // memory issues and cannot use ({0,0}, 1) ?
    int _p2[] = {0,1};
    Pattern p2 = makePatternSingleOutput(_p2, 0);
    int _p3[] = {1,1};
    Pattern p3 = makePatternSingleOutput(_p3, 1);
    int _p4[] = {1,0};
    Pattern p4 = makePatternSingleOutput(_p4, 0);

	
    Pattern patterns[] = {p1, p2, p3, p4}; // use 1,2,3,4 instead of 3,2,1,4 ...?

	// printf(" ========= length of patterns[]: %lu\n", sizeof(patterns) / sizeof(patterns[0])); 
			// only in compile time i and i is int, f is floating, need lu or unsigned long 

	int leng_pattern = (int) (sizeof(patterns) / sizeof(patterns[0]));

	printf("\n ========= length of patterns[]: %d\n", leng_pattern); 
	printf("\n ========= No of run           : %d\n", NO_OF_RUN); 
	
    // Train the network
    train_network(patterns, leng_pattern, NO_OF_RUN, nn);  
    	// 4 patterns  which is now calculated and run run 1000 times which now is NO_OF_RUN
    	// 4 and 2 meant 8 run e.g. 8 back prop ... 

	// Test the network (shall use different data but here it would be the same as it is logic)
	
    printf("\n\n nn-2-295 Testing the network mixing the build, validation and test idea due the data's nature\n"); 
    	// update pattern probably not train it I guess ?? 
    
    update_pattern(p1, nn);  // ?? p2 ... (0 0) -> 1
    for (int i=0; i < nn.n_outputs; i++) {
        printf(" ------------- pattern 001: nn.out_output[i]: %f, p2.result[i]: %i\n", nn.out_output[i], p2.result[i]);
    }
    
    update_pattern(p2, nn);  // ?? p2 ... (0 1) -> 0
    for (int i=0; i < nn.n_outputs; i++) {
        printf(" ------------- pattern 010: nn.out_output[i]: %f, p2.result[i]: %i\n", nn.out_output[i], p2.result[i]);
    }
    
    update_pattern(p3, nn);  // ?? p2 ... (1 1) -> 1
    for (int i=0; i < nn.n_outputs; i++) {
        printf(" ------------- pattern 111: nn.out_output[i]: %f, p2.result[i]: %i\n", nn.out_output[i], p2.result[i]);
    }
    
    update_pattern(p4, nn);  // ?? p2 ... (1 0) -> 0
    for (int i=0; i < nn.n_outputs; i++) {
        printf(" ------------- pattern 100: nn.out_output[i]: %f, p2.result[i]: %i\n", nn.out_output[i], p2.result[i]);
    }
    
    cudaDeviceReset();
    return 0;
}
