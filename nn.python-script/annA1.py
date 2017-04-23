# http://codereview.stackexchange.com/questions/6126/back-propagation-neural-network?rq=1

import math, time, random # , winsound
global Usefull
LearningRate = 0.001
InWeight = [[],[],[],[],[],[]]
#Generate random InWeights
for i in range(6):
    for j in range(21):
        InWeight[i].append(random.uniform(0,1))
#21 Input Values
InNeuron = [0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,
            0,0,0,0,0,0,0]
#6 Hidden Neurons
HiddenLayer = [0, 0, 0, 0, 0, 0]
#Used to calculate Delta 
HiddenLayerNoSigmoid = [0, 0, 0, 0, 0, 0]
HiddenWeight = [[],[],[]]
#Generate random HiddenWeights
for i in range(3):
    for j in range(6):
        HiddenWeight[i].append(random.uniform(0,1))
#3 Output Neurons
OutNeuron = [0, 0, 0]
#Used to calculate Delta
OutNeuronNoSigmoid = [0, 0, 0]
#Learning Table
#Engels - Nederlands - Frans - Desired output
test = [[11, 4, 8, 1, 14, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[4, 0, 6, 0, 4, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[6, 0, 6, 0, 11, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[23, 0, 0, 0, 13, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[18, 4, 4, 2, 14, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[14, 1, 6, 0, 10, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[19, 0, 2, 0, 18, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[13, 1, 1, 1, 15, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[19, 3, 1, 0, 14, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 6, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 3, 0, 7, 1, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 1, 0, 12, 7, 8, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 4, 4, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 1, 13, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 14, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 4, 9, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 6, 0, 8, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 2, 7, 0, 1, 0, 0, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 1, 0, 2, 0, 1, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 2, 2, 0, 0, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 7, 0, 2, 0, 2, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 7, 1, 1, 2, 3, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 8, 0, 2, 0, 2, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 3, 1, 3, 0, 0, 1]]
test += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 1, 5, 1, 2, 0, 0, 0, 0, 1]]

def Sigmoid(Value):
    return math.tanh(Value)

def DSigmoid(Value):  
    return 1.0 - Value**2

def UpdateHiddenNode():
    global InNeuron, InWeight
    for i in range(6):
        e = 0
        for j in range(21):
            e += InWeight[i][j]*InNeuron[j]
        HiddenLayerNoSigmoid = e
        HiddenLayer[i] = Sigmoid(e)

def UpdateOutNeuron():
    global HiddenLayer, HiddenWeight
    for i in range(3):
        e = 0
        for j in range(3):
            e += HiddenWeight[i][j]*HiddenLayer[j]
        OutNeuron[i] = Sigmoid(e)

def UpdateDelta():
    global Delta3, Delta4, Delta5, Delta6, Delta7, Delta8
    Delta3 = Delta0*HiddenWeight[0][0]+Delta1*HiddenWeight[1][0]+Delta2*HiddenWeight[2][0]
    Delta4 = Delta0*HiddenWeight[0][1]+Delta1*HiddenWeight[1][1]+Delta2*HiddenWeight[2][1]
    Delta5 = Delta0*HiddenWeight[0][2]+Delta1*HiddenWeight[1][2]+Delta2*HiddenWeight[2][2]
    Delta6 = Delta0*HiddenWeight[0][3]+Delta1*HiddenWeight[1][3]+Delta2*HiddenWeight[2][3]
    Delta7 = Delta0*HiddenWeight[0][4]+Delta1*HiddenWeight[1][4]+Delta2*HiddenWeight[2][4]
    Delta8 = Delta0*HiddenWeight[0][5]+Delta1*HiddenWeight[1][5]+Delta2*HiddenWeight[2][5]

def UpdateInWeights():
    global Delta3, Delta4, Delta5, Delta6, Delta7, Delta8
    for i in range(21):
        InWeight[0][i] += LearningRate*Delta3*DSigmoid(HiddenLayerNoSigmoid[0])*InNeuron[i]
        InWeight[1][i] += LearningRate*Delta4*DSigmoid(HiddenLayerNoSigmoid[1])*InNeuron[i]
        InWeight[2][i] += LearningRate*Delta5*DSigmoid(HiddenLayerNoSigmoid[2])*InNeuron[i]
        InWeight[3][i] += LearningRate*Delta6*DSigmoid(HiddenLayerNoSigmoid[3])*InNeuron[i]
        InWeight[4][i] += LearningRate*Delta7*DSigmoid(HiddenLayerNoSigmoid[4])*InNeuron[i]
        InWeight[5][i] += LearningRate*Delta8*DSigmoid(HiddenLayerNoSigmoid[5])*InNeuron[i]

def UpdateHiddenWeights():
    global Delta0, Delta1, Delta2
    for i in range(3):
        HiddenWeight[0][i] += LearningRate*Delta0*DSigmoid(OutNeuronNoSigmoid[0])*HiddenLayer[i]
        HiddenWeight[1][i] += LearningRate*Delta1*DSigmoid(OutNeuronNoSigmoid[1])*HiddenLayer[i]
        HiddenWeight[2][i] += LearningRate*Delta2*DSigmoid(OutNeuronNoSigmoid[2])*HiddenLayer[i]

print("Learning...")
#Start playing Learning.wav if available, else play windows default sound
#ASYNC ensures the program keeps running while playing the sound
#winsound.PlaySound("Learning.wav", winsound.SND_ASYNC)
#Start timer
StartTime = time.clock()       
Iterations = 0
#Main loop
while Iterations <= 100000:
    for i in range(len(test)):
        for j in range(21):
            InNeuron[j] = test[i][j]
        UpdateHiddenNode()
        UpdateOutNeuron()
        Delta0 = test[i][21] - OutNeuron[0]
        Delta1 = test[i][22] - OutNeuron[1]
        Delta2 = test[i][23] - OutNeuron[2]
        UpdateDelta()
        UpdateInWeights()    
        UpdateHiddenWeights()
    if Iterations % 1000 == 0:
        PercentComplete = Iterations / 1000
        print("Learning " + str(PercentComplete) + "% Complete")
    Iterations += 1
#Stop playing any sound
#winsound.PlaySound(None, winsound.SND_ASYNC)
print(Delta0, Delta1, Delta2)
#Save brain to SaveFile
SaveFileName = input("Save brain as: ")
SaveFile = open(SaveFileName+".txt", "w")
SaveFile.write(str(InWeight))
SaveFile.write(str(HiddenWeight))
SaveFile.close()
ElapsedTime = (time.clock() - StartTime)
print(str(ElapsedTime) + "seconds")
#Start playing Ready.wav if available, else play default windows sound
#ASYNC ensures the program keeps running while playing the sound
# winsound.PlaySound("Ready.wav", winsound.SND_ASYNC)

def Input_Frequency(Document):
    WantedWords = ["i", "you", "he", "are", "the", "and", "for",
                    "ik", "jij", "hij", "zijn", "het", "niet", "een",
                    "le", "tu", "il", "avez", "une", "alors", "dans"]
    file = open(Document, "r")
    text = file.read( )
    file.close()
    #Create dictionary 
    word_freq ={}
    #Split text in words
    text = str.lower(text)
    word_list = str.split(text)

    for word in word_list:
        word_freq[word] = word_freq.get(word, 0) + 1

    #Get keys 
    keys = word_freq.keys()

    #Get frequency of usefull words
    Usefull = []
    for word in WantedWords:
        if word in keys:
            word = word_freq[word]
            Usefull.append(word)
        else:
            Usefull.append(0)
    return Usefull

def UseIt(Input):
    for i in range(len(Input)):
        InNeuron[i] = Input[i]
    UpdateHiddenNode()
    UpdateOutNeuron()
    if OutNeuron[0] > 0.99:
        return ("Engelse tekst")
    if OutNeuron[1] > 0.99:
        return ("Nederlandse tekst")
    if OutNeuron[2] > 0.99:
        return ("Franse tekst")
#Documents to investigate
#Error handling checks if you input a number
while True:
    try:
        NumberOfDocuments = int(input("Aantal te onderzoeken documenten: "))
        break
    except ValueError:
        print("That was not a valid number.")
x = 0
while NumberOfDocuments > x:
    #Error handling checks if document exists
    while True:
        try:
            Document = str(input("Document: "))
            file = open(Document, "r")
            break
        except IOError:
            print(Document +" not found")
    print(UseIt(Input_Frequency(Document)))
    #Stop playing any sound
    if x == (NumberOfDocuments - 1):
        winsound.PlaySound(None, winsound.SND_ASYNC)
    x += 1
