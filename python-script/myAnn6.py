from __future__ import print_function
import numpy as np
import time

start = time.time()

# alpha = learning rate = or jump around = .5
# never update the bias hence weight 99 does nit apply to bias
#    i just override it for the value to all 1. should it be from a matrix ???
#    ??? but is the weight of bias update and if so is the formula right
#    ??? here it is strange whilst bias weight update sonehow
#    ??? it updates 0.6 and 0.35 but not 99 ... not sure why

# https://iamtrask.github.io/2015/07/27/python-network-part2/

# not sure http://www.snee.com/bobdc.blog/2016/12/a-modern-neural-network-in-11.html


# http://arduinobasics.blogspot.hk/2011/08/neural-network-part-6-back-propagation.html

# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

#   with bias but not weight update
#   both bias and weight of bias update can be done via overriding using source to the bias and weight or not



#alphas = [0.001,0.01,0.1,1,10,100,1000]
alphas = [0.5]

hiddenSize = 3 # 1 # 2 # 32

bias = True

print ("------")
print ("hiddenSize ", hiddenSize)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
#X = np.array([[0,0,1],
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])

#X = np.array([[0,0,1]])

X = np.array([[0.05,0.1,1]])



print ("X")
print ("X",X," and shape is ", X.shape)
    
                
#y = np.array([[0],
#			[1],
#			[1],
#			[0]])

#y = np.array([[0]])

y = np.array([[0.01,0.99]])

print ("y")
print ("y",y," and shape is ", y.shape)


for alpha in alphas:
    print ("\nTraining With Alpha:" , str(alpha))
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    #synapse_0 = 2*np.random.random((X.shape[1],hiddenSize)) - 1 #3
    synapse_0  = np.array([[.15, .25, 99], 
                           [.20, .30, 99],
                           [.35, .35, 99]]) # row is to neuron 1 see the below one and guess 

    synapse_1 = 2*np.random.random((hiddenSize,y.shape[1])) - 1 #2
    synapse_1 = np.array([[.4, .50],
                          [.45, .55],
                          [.6, .6]])

    print("------------------------------------------")

    print ("-- starting with alpha", alpha)
    print ("Synapse 0")
    print ("synapse_0",synapse_0," and shape is ", synapse_0.shape)
        
    print ("Synapse 1")
    print ("synapse_1",synapse_1," and shape is ", synapse_1.shape)

    for j in xrange(60000):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        if bias:
            temp = X[0][-1]
            #print ("temp",temp," and shape is ", temp.shape)
            # not working layer_1[-1] = temp replace all
            # not working np.put(layer_1,-1,temp)
            layer_1[0][-1] = X[0][-1]
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        #it seems to work --- as the last layer it does not need to update bias???
        #if bias:
        #    layer_2[0][-1] = X[0][-1]
        
        if (j in  [0,1,2]):

            print ()
            print ("round",j)
            print ("========")
            print ("layer 0")
            print ("-------")
            print ("layer_0",layer_0," and shape is ", layer_0.shape)
                
            print ()
            print ("layer 1")
            print ("-------")
            print ("layer_0",layer_0," and shape is ", layer_0.shape)
            print ("synapse_0",synapse_0," and shape is ", synapse_0.shape)
            print ("np.dot(layer_0,synapse_0)",np.dot(layer_0,synapse_0), \
                   " and shape is ", np.dot(layer_0,synapse_0).shape)
            print ("sigmoid(np.dot(layer_0,synapse_0))",sigmoid(np.dot(layer_0,synapse_0)),\
                   " and shape is ", sigmoid(np.dot(layer_0,synapse_0)).shape)
            print ("layer_1",layer_1," and shape is ", layer_1.shape)
            
            print ()
            print ("layer 2")
            print ("-------")
            print ("layer_1",layer_1," and shape is ", layer_1.shape)
            print ("synapse_1",synapse_1," and shape is ", synapse_1.shape)
            print ("np.dot(layer_1,synapse_1)",np.dot(layer_1,synapse_1), \
                   " and shape is ", np.dot(layer_1,synapse_1).shape)
            print ("sigmoid(np.dot(layer_1,synapse_1))",sigmoid(np.dot(layer_1,synapse_1)),\
                   " and shape is ", sigmoid(np.dot(layer_1,synapse_1)).shape)
            print ("layer_2",layer_2," and shape is ", layer_2.shape)
            

        # how much did we miss the target value?
        layer_2_error = layer_2 - y

        #if (j% 10000) == 0:
        #    print ("Error after ",str(j)," iterations:" , \
        #          str(np.mean(np.abs(layer_2_error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        #if synapse_1.T.shape <> (1,1):
        #    layer_1_error = layer_2_delta.dot(synapse_1.T)
        #else:
        #    layer_1_error = np.outer(layer_2_delta,synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        if (j in  [0,1,2]):

            print ("j", j, "<---")
            print ("layer_2_error = layer_2 - y")
            print ("---------------------------")
            print ("layer_2",layer_2," and shape is ", layer_2.shape)
            print ("y",y," and shape is ", y.shape)
            print ("layer_2_error",layer_2_error," and shape is ", layer_2_error.shape)
                
            print ()
            print ("layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)")
            print ("--------------------------------------------------------------------")
            print ("layer_2_error",layer_2_error," and shape is ", layer_2_error.shape)
            print ("sigmoid_output_to_derivative(layer_2)",sigmoid_output_to_derivative(layer_2)," and shape is ", \
                   sigmoid_output_to_derivative(layer_2).shape)
            print ("layer_2_delta",layer_2_delta," and shape is ", layer_2_delta.shape)

            print ()
            print ("layer_1_error = layer_2_delta.dot(synapse_1.T)")
            print ("----------------------------------------------")
            print ("synapse_1.T",synapse_1.T," and shape is ", synapse_1.T.shape)
            print ("layer_2_delta",layer_2_delta," and shape is ", layer_2_delta.shape)
            print ("layer_1_error",layer_1_error," and shape is ", layer_1_error.shape)

            print ("layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)")
            print ("layer_1",layer_1," and shape is ", layer_1.shape)
            print ("sigmoid_output_to_derivative(layer_1)",sigmoid_output_to_derivative(layer_1),"\n and shape is ", \
                   sigmoid_output_to_derivative(layer_1).shape)            
            print ("layer_1_error",layer_1_error," and shape is ", layer_1_error.shape)
            print ("layer_1_delta",layer_1_delta," and shape is ", layer_1_delta.shape)

            print ("Synapse 0 before ")
            print ("synapse_0 before ",synapse_0," and shape is ", synapse_0.shape)
            print ("Synapse 1 before ")
            print ("synapse_1 before ",synapse_1," and shape is ", synapse_1.shape)
        
        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        #if layer_1.T.shape <> (1,1):
        #    synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        #else:
        #    synapse_1 -= alpha * np.outer(layer_1.T,layer_2_delta)

        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
        #if layer_0.T.shape <> (1,1):
        #    synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
        #else:
        #    synapse_0 -= alpha * np.outer(layer_0.T,layer_1_delta)
        
        if (j in  [0,1,2]):

            print ("Synapse 0 after ")
            print ("synapse_0 after ",synapse_0," and shape is ", synapse_0.shape)
            print ("Synapse 1 after ")
            print ("synapse_1 after ",synapse_1," and shape is ", synapse_1.shape)
            print("********************************************")
            print()
            end = time.time()
            print("during from start:", end - start)


    print ("Error 60000 iterations:" , str(np.mean(np.abs(layer_2_error))))

    print ("-- ending with alpha", alpha)

    print ("layer 0")
    print (layer_0,"\n and shape is ", layer_0.shape)
        
    print ("layer 1")
    print (layer_1,"\n and shape is ", layer_1.shape)

    print ("layer 2")
    print (layer_2,"\n and shape is ", layer_2.shape)

    print ("Synapse 0")
    print (synapse_0,"\n and shape is ", synapse_0.shape)
        
    print ("Synapse 1")
    print (synapse_1,"\n and shape is ", synapse_1.shape)
    
    end = time.time()
    print("during from start:", end - start)
    
    print ("================ completed ======================")