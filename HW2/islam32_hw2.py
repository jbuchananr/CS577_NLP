import numpy as np
import string
import argparse
import sys
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from random import randrange


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.autograd import Variable


torch.manual_seed(1)


import re



'''Following are some helper functions from https://github.com/lixin4ever/E2E-TBSA/blob/master/utils.py to help parse the Targeted Sentiment Twitter dataset. You are free to delete this code and parse the data yourself, if you wish.

You may also use other parsing functions, but ONLY for parsing and ONLY from that file.
'''
def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')
                if tag == 'O':
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset



##################......option 3.......###########

class Net_3(nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        #self.lstm = nn.LSTM(30, 10 // 2, num_layers=1, bidirectional=True) 
        self.lstm = nn.LSTM(304, 150 // 2, num_layers=1, bidirectional=True)
        #self.fc1 = nn.Linear(20, 12)
        self.fc2 = nn.Linear(150, 4) 

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1)) 
        # print (x, x.size())
        #print('resize', x.view(1,1,-1), x.view(len(x),1,-1).size()) #torch.Size([90, 1, 20])
        lstm_out, _ = self.lstm(x.view(len(x),1,-1)) 
        #print('lstm_out', lstm_out, lstm_out.size()) # torch.Size([90, 1, 12])
        #tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        x = self.fc2(lstm_out.view(len(x), -1))
        #print(x, x.size()) #torch.Size([90, 4])
        out = F.log_softmax(x, dim = 1)
        return out

def create_nn_option3(batch_size,opt, learning_rate, epochs,trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim):
    #DIM=21
    #DIM=300
    dimensions=(dimensionsOrg*1) + tag_dim
    trainModelFlag = True
    if trainModelFlag:
        net = Net_3()
        #print(net)
        # create a stochastic gradient descent optimizer
        if(opt=="SGD"):
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        elif(opt=="ADA"):
            optimizer = optim.Adadelta(net.parameters(), lr=1.0, eps=1e-06, weight_decay=0)
        elif(opt=="ADAM"):
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
        # create a loss function
        criterion = nn.NLLLoss()
        batch_sizeOrg=batch_size
        # run the main training loop
        for epoch in range(epochs):
            batch_size=batch_sizeOrg
            for i in range(0,len(trainTuplesConcat),batch_size):
                #print (i)
                if(i+batch_size > len(trainTuplesConcat)):
                    batch_size = len(trainTuplesConcat) - i
                optimizer.zero_grad()
                #print("batch_size",batch_size) #10000
                #print ("trainTuplesConcat",trainTuplesConcat, trainTuplesConcat.size()) #torch.Size([36673, 21])
                #data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data,requires_grad=True)
                data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data.view(batch_size,dimensions),requires_grad=True)
                
                #print('data', data, data.size()) #torch.Size([10000, 21])
                # target= autograd.Variable(targetTuplesConcat[i:(i+batch_size)].data)
                target= targetTuplesConcatIndex[i:(i+batch_size)]
                # print target
                # target_keys=[]
                # for k in range(batch_size):
                #     target_keys.append(get_key(target[k],label_embeddings))
                target_keys=target
                exp=autograd.Variable(torch.LongTensor(target_keys))
                loss= criterion(net(data), exp)
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i, len(trainTuplesConcat),
                               100. * i / len(trainTuplesConcat), loss.item()))
        torch.save(net.state_dict(),"neural_net_modelADAM_option3.pt")

def trainNeuralNet_option3(trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim):
    opt="ADAM"
    #opt="ADA"
    #batch_size=10000
    #dimensions=10
    #batch_size=5239
    #batch_size=2000
    batch_size=100
    
    learning_rate=0.05
    #learning_rate=0.1
    epochs=100
    #log_interval=1000
    #trainModelFlag= True
    create_nn_option3(batch_size, opt,learning_rate, epochs,trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim)

def startTraining_option3(tag_to_ix,label_embeddings,sentence_in, targets, dimensionsOrg, tag_dim):
    createTotalEmbedding=True
    if(createTotalEmbedding):
        for i in range(len(sentence_in)):
            #print (train_lex[i])
            for j in range(len(sentence_in[i])):
                #concat word and label
                #print (train_lex[i][j])
                #sys.exit()
                if j==0:
                    #concat first word with prev label IE START Label
                    #if first word then put start label as the previous word
                    trainTuple=torch.cat(( sentence_in[i][j].view(1, 300),label_embeddings[4]),1)
                    #trainTuple=torch.cat(( word_embeddings[train_lex[i][j]],label_embeddings[4],label_embeddings[4]),1)
                    #print(trainTuple)
                    if i==0:
                        trainTuplesConcat=trainTuple
                        # print(label_embeddings)
                        # print(targets[i][j])
                        targetTuplesConcat=label_embeddings[targets[i][j]]
                        targetTuplesConcatIndex=torch.LongTensor([targets[i][j]])
                    else: 
                        trainTuplesConcat=torch.cat((trainTuplesConcat,trainTuple),0)
                        targetTuplesConcat=torch.cat((targetTuplesConcat,label_embeddings[targets[i][j]]),0)
                        targetTuplesConcatIndex=torch.cat((targetTuplesConcatIndex,torch.LongTensor([targets[i][j]])),0)
                else:
                    trainTuple=torch.cat(( sentence_in[i][j].view(1, 300),label_embeddings[targets[i][j-1]]),1)
                    # print trainTuple
                    trainTuplesConcat=torch.cat((trainTuplesConcat,trainTuple),0)
                    targetTuplesConcat=torch.cat((targetTuplesConcat,label_embeddings[targets[i][j]]),0)
                    targetTuplesConcatIndex=torch.cat((targetTuplesConcatIndex,torch.torch.LongTensor([targets[i][j]])),0)
        trainNeuralNet_option3(trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim)

def viterbi_option3(dimensionsOrg, tag_dim, idx2label,label_embeddings,valid_lex,valid_y):
    net=Net_3()
    net.load_state_dict(torch.load("neural_net_modelADAM_option3.pt"))
    #net.load_state_dict(torch.load("neural_net_modelADAM4_option2_tag_dim_wofeature_1layer.pt"))
    criterion = nn.NLLLoss()
    dimensions=(dimensionsOrg*1) +tag_dim
    totalOutput=[]
    rows=len(label_embeddings)-1
    for i in range(len(valid_lex)):
        # print i
        #FORWARD PASS TO CREATE DP TABLE
        cols=len(valid_lex[i])
        #-1 to account for start label
        viterbiProbTable = np.zeros(shape=(rows,cols))
        viterbiBackBackTable = np.zeros(shape=(rows,cols))
        for j in range(cols):
            if(j==0):
                #if first word then prev label is only start label
                prev_label=(label_embeddings[4])
                word_embed=torch.cat((valid_lex[i][j].view(1, 300),prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                #print(data, data.size())
                prediction = net(data)
                # print prediction.data.view(127,1).np().shape
                #print("prediction", prediction, prediction.size())  # torch.Size([1, 7])
                colProb=prediction.data.view(4).numpy()
                #print ("colProb", colProb)
                viterbiProbTable[:,j]=colProb
                viterbiBackBackTable[:,j]=5
                
            elif(j!=0):
                for k in range(rows):
                    prev_label=(label_embeddings[k])
                    word_embed=torch.cat((valid_lex[i][j].view(1, 300),prev_label),1)
                    data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                    prediction = net(data)
                    colProb=prediction.data.view(4).numpy()
                    if k==0:
                        viterbiProbTable[:,j]=colProb+viterbiProbTable[k][j-1]
                        viterbiBackBackTable[:,j]=k
                    else:
                        for x in range(rows):
                            if(viterbiProbTable[x][j]<colProb[x]+viterbiProbTable[k][j-1]):
                                viterbiProbTable[x][j]=colProb[x]+viterbiProbTable[k][j-1]
                                viterbiBackBackTable[x][j]=k
        # print viterbiProbTable
        # print viterbiBackBackTable
        #BACKWARD PASS TO CREATE PATH
        output=[]
        for j in range(cols-1,-1,-1):
            if j==cols-1:
                row_index = viterbiProbTable[:,j].argmax(axis=0)
                output.append(row_index)
                prevLabel=viterbiBackBackTable[row_index][j]
                # print prevLabel
            else:
                output.append(int(prevLabel))
                # print viterbiBackBackTable[int(prevLabel)][j]
                prevLabel=viterbiBackBackTable[int(prevLabel)][j]
        output.reverse()
        #print (output,valid_y[i], len(output), len(valid_y[i]))
        #predictions_test = map(lambda t: idx2label[t], output)
        #print (predictions_test)
        #totalOutput.append(predictions_test)
        totalOutput.append(output)
    #print("totalOutput", totalOutput, len(totalOutput))# 235
    return totalOutput

def predict_with_neural_net_option3(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets):
    net=Net_3()
    net.load_state_dict(torch.load("neural_net_modelADAM_option3.pt"))
    #net.load_state_dict(torch.load("neural_net_modelADAM4_option2_tag_dim_wofeature_1layer.pt"))
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    #print('w2v')
    dimensions=(dimensionsOrg*1)+tag_dim
    
    total_output = []
    for i in range(len(sentence_in)):
        output=[]
        for j in range(len(sentence_in[i])):
            #if(j==0 or j ==1):
            if(j==0):
                #print('j', j)
                #if first word then prev label is only start label
                prev_label=(label_embeddings[4])
                #print("prev_label",j,prev_label)
                #word_embed=torch.cat((sentence_in[i][j],word_embeddings[sentence_in[i][j]], prev_label),1)
                word_embed=torch.cat((sentence_in[i][j].view(1, 300),prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                #print(data, data.size())
                logit = net(data)
                prediction= torch.argmax(logit, dim =1 )
                #val,idx = prediction.max(0, keepdim=True)
                output.append(prediction)
                
            elif(j!=0): 
                prev_label= label_embeddings[targets[i][j-1]]
                #prev_prev_label = label_embeddings[targets[i][j-2]]
                #print("prev_label",j,prev_label)
                #print("prev_prev_label",j,prev_prev_label)
                #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                word_embed=torch.cat((sentence_in[i][j].view(1, 300),prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                logit = net(data)
                prediction= torch.argmax(logit, dim =1 )
                #val,idx = prediction.max(0, keepdim=True)
                output.append(prediction)
        total_output.append(output)
    return(total_output)
    
#################.......option 2.............############
def prepare_sequence_word_em(seq, to_ix):
    #idxs = [to_ix[w] for w in seq]
    #print type(idxs)
    idxs = []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix["my_unknown"])         
        
    return torch.tensor(idxs, dtype=torch.long)

def create_word_embeddings_w2v(weights):
    word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(weights))
    return word_embeddings

def build_vocab_using_word_embedding(train_set, wv_from_bin):
    word_to_ix = {}
    for i in train_set:
        for j in i['words']:
            if j not in word_to_ix:
                if j in wv_from_bin.vocab:
                    word_to_ix[j] = wv_from_bin.vocab[j].index
                    #print "word_to_ix", word_to_ix
    return word_to_ix

class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.fc1 = nn.Linear(304, 150) #10 dim of input, 5 for word, 5 for label, 150 size of hidden
        self.fc2 = nn.Linear(150, 4) # hidden layer to 127 output labels
        #self.fc3 = nn.Linear(100, 4)
        #self.fc4 = nn.Linear(100, 4)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #x = torch.sigmoid(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        #return F.log_softmax(x, dim = 1)
        out = F.log_softmax(x, dim = 1)
        #out = viterbi_layer(x)
        return out
def create_nn_option2(batch_size,opt, learning_rate, epochs,trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim):
    #DIM=21
    #DIM=300
    dimensions=(dimensionsOrg*1) + tag_dim
    trainModelFlag = True
    if trainModelFlag:
        net = Net_2()
        #print(net)
        # create a stochastic gradient descent optimizer
        if(opt=="SGD"):
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        elif(opt=="ADA"):
            optimizer = optim.Adadelta(net.parameters(), lr=1.0, eps=1e-06, weight_decay=0)
        elif(opt=="ADAM"):
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
        # create a loss function
        criterion = nn.NLLLoss()
        batch_sizeOrg=batch_size
        # run the main training loop
        for epoch in range(epochs):
            batch_size=batch_sizeOrg
            for i in range(0,len(trainTuplesConcat),batch_size):
                #print (i)
                if(i+batch_size > len(trainTuplesConcat)):
                    batch_size = len(trainTuplesConcat) - i
                optimizer.zero_grad()
                #print("batch_size",batch_size) #10000
                #print ("trainTuplesConcat",trainTuplesConcat, trainTuplesConcat.size()) #torch.Size([36673, 21])
                #data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data,requires_grad=True)
                data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data.view(batch_size,dimensions),requires_grad=True)
                
                #print('data', data, data.size()) #torch.Size([10000, 21])
                # target= autograd.Variable(targetTuplesConcat[i:(i+batch_size)].data)
                target= targetTuplesConcatIndex[i:(i+batch_size)]
                # print target
                # target_keys=[]
                # for k in range(batch_size):
                #     target_keys.append(get_key(target[k],label_embeddings))
                target_keys=target
                exp=autograd.Variable(torch.LongTensor(target_keys))
                loss= criterion(net(data), exp)
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i, len(trainTuplesConcat),
                               100. * i / len(trainTuplesConcat), loss.item()))
        torch.save(net.state_dict(),"neural_net_modelADAM_option2.pt")

def trainNeuralNet_option2(trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim):
    opt="ADAM"
    
    batch_size=100
    
    learning_rate=0.05
    #learning_rate=0.1
    epochs=200
    #log_interval=1000
    #trainModelFlag= True
    create_nn_option2(batch_size, opt,learning_rate, epochs,trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim)

def startTraining_option2(tag_to_ix,label_embeddings,sentence_in, targets, dimensionsOrg, tag_dim):
    createTotalEmbedding=True
    if(createTotalEmbedding):
        for i in range(len(sentence_in)):
            #print (train_lex[i])
            for j in range(len(sentence_in[i])):
                #concat word and label
                #print (train_lex[i][j])
                #sys.exit()
                if j==0:
                    #concat first word with prev label IE START Label
                    #if first word then put start label as the previous word
                    trainTuple=torch.cat(( sentence_in[i][j].view(1, 300),label_embeddings[4]),1)
                    #trainTuple=torch.cat(( word_embeddings[train_lex[i][j]],label_embeddings[4],label_embeddings[4]),1)
                    #print(trainTuple)
                    if i==0:
                        trainTuplesConcat=trainTuple
                        # print(label_embeddings)
                        # print(targets[i][j])
                        targetTuplesConcat=label_embeddings[targets[i][j]]
                        targetTuplesConcatIndex=torch.LongTensor([targets[i][j]])
                    else: 
                        trainTuplesConcat=torch.cat((trainTuplesConcat,trainTuple),0)
                        targetTuplesConcat=torch.cat((targetTuplesConcat,label_embeddings[targets[i][j]]),0)
                        targetTuplesConcatIndex=torch.cat((targetTuplesConcatIndex,torch.LongTensor([targets[i][j]])),0)
                else:
                    trainTuple=torch.cat(( sentence_in[i][j].view(1, 300),label_embeddings[targets[i][j-1]]),1)
                    # print trainTuple
                    trainTuplesConcat=torch.cat((trainTuplesConcat,trainTuple),0)
                    targetTuplesConcat=torch.cat((targetTuplesConcat,label_embeddings[targets[i][j]]),0)
                    targetTuplesConcatIndex=torch.cat((targetTuplesConcatIndex,torch.torch.LongTensor([targets[i][j]])),0)
        trainNeuralNet_option2(trainTuplesConcat,targetTuplesConcat,targetTuplesConcatIndex, dimensionsOrg, tag_dim)

def viterbi_option2(dimensionsOrg, tag_dim, idx2label,label_embeddings,valid_lex,valid_y):
    net=Net_2()
    net.load_state_dict(torch.load("neural_net_modelADAM_option2.pt"))
    #net.load_state_dict(torch.load("neural_net_modelADAM4_option2_tag_dim_wofeature_1layer.pt"))
    criterion = nn.NLLLoss()
    dimensions=(dimensionsOrg*1) +tag_dim
    totalOutput=[]
    rows=len(label_embeddings)-1
    for i in range(len(valid_lex)):
        # print i
        #FORWARD PASS TO CREATE DP TABLE
        cols=len(valid_lex[i])
        #-1 to account for start label
        viterbiProbTable = np.zeros(shape=(rows,cols))
        viterbiBackBackTable = np.zeros(shape=(rows,cols))
        for j in range(cols):
            if(j==0):
                #if first word then prev label is only start label
                prev_label=(label_embeddings[4])
                word_embed=torch.cat((valid_lex[i][j].view(1, 300),prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                #print(data, data.size())
                prediction = net(data)
                # print prediction.data.view(127,1).np().shape
                #print("prediction", prediction, prediction.size())  # torch.Size([1, 7])
                colProb=prediction.data.view(4).numpy()
                #print ("colProb", colProb)
                viterbiProbTable[:,j]=colProb
                viterbiBackBackTable[:,j]=5
                
            elif(j!=0):
                for k in range(rows):
                    prev_label=(label_embeddings[k])
                    word_embed=torch.cat((valid_lex[i][j].view(1, 300),prev_label),1)
                    data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                    prediction = net(data)
                    colProb=prediction.data.view(4).numpy()
                    if k==0:
                        viterbiProbTable[:,j]=colProb+viterbiProbTable[k][j-1]
                        viterbiBackBackTable[:,j]=k
                    else:
                        for x in range(rows):
                            if(viterbiProbTable[x][j]<colProb[x]+viterbiProbTable[k][j-1]):
                                viterbiProbTable[x][j]=colProb[x]+viterbiProbTable[k][j-1]
                                viterbiBackBackTable[x][j]=k
        # print viterbiProbTable
        # print viterbiBackBackTable
        #BACKWARD PASS TO CREATE PATH
        output=[]
        for j in range(cols-1,-1,-1):
            if j==cols-1:
                row_index = viterbiProbTable[:,j].argmax(axis=0)
                output.append(row_index)
                prevLabel=viterbiBackBackTable[row_index][j]
                # print prevLabel
            else:
                output.append(int(prevLabel))
                # print viterbiBackBackTable[int(prevLabel)][j]
                prevLabel=viterbiBackBackTable[int(prevLabel)][j]
        output.reverse()
        #print (output,valid_y[i], len(output), len(valid_y[i]))
        #predictions_test = map(lambda t: idx2label[t], output)
        #print (predictions_test)
        #totalOutput.append(predictions_test)
        totalOutput.append(output)
    #print("totalOutput", totalOutput, len(totalOutput))# 235
    return totalOutput

def predict_with_neural_net_option2(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets):
    net=Net_2()
    net.load_state_dict(torch.load("neural_net_modelADAM_option2.pt"))
    #net.load_state_dict(torch.load("neural_net_modelADAM4_option2_tag_dim_wofeature_1layer.pt"))
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    #print('w2v')
    dimensions=(dimensionsOrg*1)+tag_dim
    
    total_output = []
    for i in range(len(sentence_in)):
        output=[]
        for j in range(len(sentence_in[i])):
            #if(j==0 or j ==1):
            if(j==0):
                #print('j', j)
                #if first word then prev label is only start label
                prev_label=(label_embeddings[4])
                #print("prev_label",j,prev_label)
                #word_embed=torch.cat((sentence_in[i][j],word_embeddings[sentence_in[i][j]], prev_label),1)
                word_embed=torch.cat((sentence_in[i][j].view(1, 300),prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                #print(data, data.size())
                logit = net(data)
                prediction= torch.argmax(logit, dim =1 )
                #val,idx = prediction.max(0, keepdim=True)
                output.append(prediction)
                
            elif(j!=0): 
                prev_label= label_embeddings[targets[i][j-1]]
                #prev_prev_label = label_embeddings[targets[i][j-2]]
                #print("prev_label",j,prev_label)
                #print("prev_prev_label",j,prev_prev_label)
                #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                word_embed=torch.cat((sentence_in[i][j].view(1, 300),prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                logit = net(data)
                prediction= torch.argmax(logit, dim =1 )
                #val,idx = prediction.max(0, keepdim=True)
                output.append(prediction)
        total_output.append(output)
    return(total_output)
                      

#################.......option 1.............############



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    # idxs = []
    # for w in seq:
    #     print (w)
    #     print (to_ix[w])
    #     idxs.append(to_ix[w])
    #print ("idxs", idxs)
    return idxs
    #return torch.tensor(idxs, dtype=torch.long)


def build_index_vocab(train_set):
    word_to_ix = {}
    for i in train_set:
        for j in i['words']:
            #print (j)
            if j not in word_to_ix:
                word_to_ix[j] = len(word_to_ix)
    return word_to_ix

def get_embeddings(idx2Text,vocab_size, dimensions):
    embeds = nn.Embedding(vocab_size, dimensions)  #  words in vocab, 7 dimensional embeddings
    #print(embeds)
    values = idx2Text.values()
    #print(values)
    id2embedding = {}
    for i in values:
        #print (i)
        #id2embedding[i]=embeds(autograd.Variable(torch.LongTensor([i])))
        id2embedding[i]=embeds(torch.LongTensor([i]))


    #print (id2embedding)
    return id2embedding

def create_word_embeddings(idx2word,dimensions):
    vocab_size = len(idx2word)
    word_embeddings=get_embeddings(idx2word,vocab_size,dimensions)
    #torch.save(word_embeddings, 'wordEmbeddings.pt')
    return word_embeddings

def create_label_embeddings(idx2label,dimensions):
    labels_size = len(idx2label)
    label_embeddings=get_embeddings(idx2label,labels_size,dimensions)
    #torch.save(label_embeddings, 'labelEmbeddings.pt')
    return label_embeddings


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(50, 100) #10 dim of input, 5 for word, 5 for label, 150 size of hidden
        self.fc2 = nn.Linear(100, 4) # hidden layer to 127 output labels
        #self.fc3 = nn.Linear(200, 4) 
        #self.fc4 = nn.Linear(100, 4) 
    def forward(self, x):
        x = self.fc1(x)
        #print('x1', x)
        #x = F.relu(x)
        x = F.relu(x)
        #print('h1', x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        #print('x2', x)
        # x = F.relu(x)
        # x = self.fc3(x)
        #return F.log_softmax(x, dim = 1)
        #out = F.log_softmax(x3, dim = 1)
        #out = torch.sigmoid(x)
        #out = F.relu(x)
        #print('out', out)
        return x 
def create_nn(batch_size,opt, learning_rate, epochs,trainTuplesConcat,targetTuplesConcat, targetTuplesConcatIndex, dimensionsOrg):
    dimensions=dimensionsOrg*5
    trainModelFlag = True
    if trainModelFlag:
        net = Net()
        #print(net)
        # create a stochastic gradient descent optimizer
        if(opt=="SGD"):
            #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        elif(opt=="ADA"):
            optimizer = optim.Adadelta(net.parameters(), lr=learning_rate, eps=1e-06, weight_decay=.1)
        elif(opt=="ADAM"):
            optimizer = optim.Adam(net.parameters(), lr=learning_rate,  weight_decay=0.0001)
        # create a loss function
        #criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()

        batch_sizeOrg=batch_size
        # run the main training loop
        for epoch in range(epochs):
            batch_size=batch_sizeOrg
            for i in range(0,len(trainTuplesConcat),batch_size):
                #print (i)
                if(i+batch_size > len(trainTuplesConcat)):
                    batch_size = len(trainTuplesConcat) - i
                optimizer.zero_grad()
                #print("batch_size",batch_size) #10000
                #print ("trainTuplesConcat",trainTuplesConcat, len(trainTuplesConcat)) #torch.Size([36673, 21]) #2115
                #print ("trainTuplesConcat",trainTuplesConcat[i:(i+batch_size)], len(trainTuplesConcat[i:(i+batch_size)])) #1000
                
                #data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data,requires_grad=True) #my
                data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data.view(batch_size,dimensions),requires_grad=True)
                
                #data = trainTuplesConcat[i:(i+batch_size)]
                #print(data, type(data), data.size())
                #print('data', data, len(data), len(data[0]), len(data[1]), len(data[0][0]),type(data), type(data[0]),  type(data[0][0])) #1000 17 7 1 <class 'list'> <class 'list'>  <class 'torch.Tensor'>
                pred = net(data)
                #print('pred', pred, pred.size())
                
                target= autograd.Variable(targetTuplesConcatIndex[i:(i+batch_size)].data)
                #target= targetTuplesConcatIndex[i:(i+batch_size)]
                #print ('target',target, target.size())
                # target_keys=target
                # true =autograd.Variable(torch.LongTensor(target_keys))
                # print(true)
                
                loss= criterion(pred, target)
                #print('loss', loss)
                #sys.exit()
                loss.backward(retain_graph=True)
                optimizer.step()
                # pert = pred.grad.sign() * learning_rate
                # data = torch.clamp(data + pert, 0, 1)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i, len(trainTuplesConcat),
                               100. * i / len(trainTuplesConcat), loss.item()))
                #logit = F.log_softmax(pred, dim =1 )
                # prediction= torch.argmax(logit, dim =1 )
                # print(prediction, prediction.size()) ###torch.Size([240])
                #prediction= torch.argmax(logit, dim =0 )
                #print(prediction, prediction.size())
                #val,idx = prediction.max(0, keepdim=True)
                #print(val)
                #print(idx)
                
        torch.save(net.state_dict(),"neural_net_modelADAM_1.pt")



#def trainNeuralNet(trainTuplesConcat,targetTuplesConcat,word_embeddings,label_embeddings,targetTuplesConcatIndex):
def trainNeuralNet(trainTuplesConcat,targetTuplesConcat, targetTuplesConcatIndex, dimensionsOrg):

    opt="ADAM"
    #opt ="SGD"
    #opt ="ADA"


    
    batch_size=64
    
    #learning_rate=0.01
    #learning_rate=0.1
    learning_rate=0.05
    #learning_rate=0.001
    epochs= 200
    
    #create_nn(batch_size, opt, dimensions ,learning_rate, epochs,log_interval,trainTuplesConcat,targetTuplesConcat,word_embeddings,label_embeddings,trainModelFlag,targetTuplesConcatIndex)
    create_nn(batch_size, opt,learning_rate, epochs,trainTuplesConcat,targetTuplesConcat, targetTuplesConcatIndex, dimensionsOrg)
###feature extraction
def startTraining(word_embeddings,label_embeddings,train_x,train_y, dimensionsOrg):
    createTotalEmbedding=True
    if(createTotalEmbedding):
        for i in range(len(train_x)):
            # print train_lex[i], map(lambda t: idx2word[t], train_lex[i])
            # print train_y[i], map(lambda t: idx2label[t], train_y[i])
            # print "\n"
            for j in range(len(train_x[i])):
                #concat word and label
                if j==0 : ### each row's first column
                #if j==0 or j==1: 
                    ####if first word then put start label as the previous label (<start>)
                    current_word = word_embeddings[train_x[i][j]]
                    prev_word = word_embeddings[train_x[i][j]]
                    prev_prev_word = word_embeddings[train_x[i][j]]
                    current_tag = label_embeddings[train_y[i][j]]
                    prev_tag  = label_embeddings[4]
                    prev_prev_tag  = label_embeddings[4]
                    ##### concate feature ###
                    context =torch.cat(( current_word, prev_prev_word, prev_word, prev_prev_tag, prev_tag),1)
                    if i == 0:
                        all_context = context
                        all_target = current_tag
                        all_target_index = torch.LongTensor([train_y[i][j]])
                    
                    else: 
                        all_context=torch.cat((all_context, context),0)
                        all_target=torch.cat((all_target,current_tag),0)
                        all_target_index=torch.cat((all_target_index,torch.LongTensor([train_y[i][j]])),0) 
                
                if j==1: ### each row's second column
                    current_word = word_embeddings[train_x[i][j]]
                    prev_word = word_embeddings[train_x[i][j-1]]
                    prev_prev_word = word_embeddings[train_x[i][j-1]]
                    current_tag = label_embeddings[train_y[i][j]]
                    prev_tag  = label_embeddings[train_y[i][j-1]]              
                    prev_prev_tag  = label_embeddings[train_y[i][j-1]]  
                    ##### concate feature ###
                    context =torch.cat(( current_word, prev_prev_word, prev_word, prev_prev_tag, prev_tag),1)
                    #print(context, context.size()) #torch.Size([1, 30])
                    #sys.exit()
                    all_context=torch.cat((all_context, context),0)
                    all_target=torch.cat((all_target,current_tag),0)
                    all_target_index=torch.cat((all_target_index,torch.LongTensor([train_y[i][j]])),0) 

                elif(j!=0 and j!= 1): ### each row's other columns except first and second column

                    current_word = word_embeddings[train_x[i][j]]
                    prev_word = word_embeddings[train_x[i][j-1]]
                    prev_prev_word = word_embeddings[train_x[i][j-2]]
                    current_tag = label_embeddings[train_y[i][j]]
                    prev_tag  = label_embeddings[train_y[i][j-1]]
                    prev_prev_tag  = label_embeddings[train_y[i][j-2]]  
                    ##### concate feature ###
                    #context=torch.cat(( current_word, prev_word, prev_tag),1)
                    context =torch.cat(( current_word, prev_prev_word, prev_word, prev_prev_tag, prev_tag),1)
                    #print(context, context.size()) #torch.Size([1, 30])
                    #sys.exit()
                    all_context=torch.cat((all_context, context),0)
                    all_target=torch.cat((all_target,current_tag),0)
                    all_target_index=torch.cat((all_target_index,torch.LongTensor([train_y[i][j]])),0) 
        trainNeuralNet(all_context, all_target, all_target_index, dimensionsOrg)


def predict_with_neural_net(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets):
    net=Net()
    net.load_state_dict(torch.load("neural_net_modelADAM_1.pt"))
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    #dimensions=dimensionsOrg*3
    dimensions=dimensionsOrg*5
    total_output = []
    for i in range(len(sentence_in)):
        output=[]
        for j in range(len(sentence_in[i])):
            #if(j==0 or j ==1):
            if(j==0):
                #print('j', j)
                #if first word then prev label is only start label
                prev_label=(label_embeddings[4])
                #print("prev_label",j,prev_label)
                #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j]],prev_label),1)
                word_embed=torch.cat((word_embeddings[sentence_in[i][j]],word_embeddings[sentence_in[i][j]],word_embeddings[sentence_in[i][j]],prev_label, prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                #print(data, data.size())
                logit = F.log_softmax(net(data), dim = 1)
                prediction= torch.argmax(logit, dim =0 )
                val,idx = prediction.max(0, keepdim=True)
                output.append(idx)
            if(j==1):
                prev_label= label_embeddings[targets[i][j-1]]
                prev_prev_label = label_embeddings[4]
                #print("prev_label",j,prev_label)
                #print("prev_prev_label",j,prev_prev_label)
                #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                word_embed=torch.cat((word_embeddings[sentence_in[i][j]],word_embeddings[sentence_in[i][j-1]],word_embeddings[sentence_in[i][j-1]], prev_prev_label, prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                logit = F.log_softmax(net(data), dim = 1)
                prediction= torch.argmax(logit, dim =0 )
                val,idx = prediction.max(0, keepdim=True)
                output.append(idx)
                
            elif(j!=0 and j!= 1): 
                prev_label= label_embeddings[targets[i][j-1]]
                prev_prev_label = label_embeddings[targets[i][j-2]]
                #print("prev_label",j,prev_label)
                #print("prev_prev_label",j,prev_prev_label)
                #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                word_embed=torch.cat((word_embeddings[sentence_in[i][j]],word_embeddings[sentence_in[i][j-2]],word_embeddings[sentence_in[i][j-1]],prev_prev_label,prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                logit = F.log_softmax(net(data), dim = 1)
                prediction= torch.argmax(logit, dim =0 )
                val,idx = prediction.max(0, keepdim=True)
                output.append(idx)
        total_output.append(output)
    return(total_output)
                      


def viterbi(dimensionsOrg,idx2word,idx2label,word_embeddings,label_embeddings,valid_lex,valid_y):
    net=Net()
    net.load_state_dict(torch.load("neural_net_modelADAM_1.pt"))
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    #dimensions=dimensionsOrg*3
    dimensions=dimensionsOrg*5
    totalOutput=[]
    rows=len(label_embeddings)-1
    for i in range(len(valid_lex)):
        # print i
        #FORWARD PASS TO CREATE DP TABLE
        cols=len(valid_lex[i])
        #-1 to account for start label
        viterbiProbTable = np.zeros(shape=(rows,cols))
        viterbiBackBackTable = np.zeros(shape=(rows,cols))
        #print(viterbiBackBackTable)
        for j in range(cols):
            #if(j==0 or j ==1):
            if(j==0):
                #print('j', j)
                #if first word then prev label is only start label
                prev_label=(label_embeddings[4])
                #print("prev_label",j,prev_label)
                #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j]],prev_label),1)
                word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j]],prev_label, prev_label),1)
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                #print(data, data.size())
                prediction = F.log_softmax(net(data), dim = 1)
                # print prediction.data.view(127,1).np().shape
                #print("prediction", prediction, prediction.size())  # torch.Size([1, 7])
                colProb=prediction.data.view(4).numpy() #conver tensor to numpy array
                #print ("colProb", colProb)
                #print("viterbiProbTable[:,j]",viterbiProbTable[:,j])
                viterbiProbTable[:,j]=colProb
                #print('viterbiProbTable', viterbiProbTable)
                viterbiBackBackTable[:,j]=5
                #print('viterbiBackBackTable',j, viterbiBackBackTable)
                
            if(j==1):
                for k in range(rows):
                    #prev_label= label_embeddings[k]
                    prev_label= label_embeddings[valid_y[i][j-1]]
                    prev_prev_label = label_embeddings[4]
                    #print("prev_label",j,prev_label)
                    #print("prev_prev_label",j,prev_prev_label)
                    #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                    word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],word_embeddings[valid_lex[i][j-1]], prev_prev_label, prev_label),1)
                    data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                    # logit = F.log_softmax(pred, dim =1 )
                    # # prediction= torch.argmax(logit, dim =1 )
                    # # print(prediction, prediction.size()) ###torch.Size([240])
                    # prediction= torch.argmax(logit, dim =0 )
                    # print(prediction, prediction.size())
                    # val,idx = prediction.max(0, keepdim=True)
                    # #print(val)
                    # print(idx)
                    
                    prediction = F.log_softmax(net(data), dim =1 )
                    pred= torch.argmax(prediction, dim =0 )
                    val,idx = pred.max(0, keepdim=True)
                    #print(idx)
                    colProb=prediction.data.view(4).numpy()
                    # print(colProb)
                    # print(colProb[1])
                    if k==0:
                        viterbiProbTable[:,j]=colProb+viterbiProbTable[k][j-1]
                        #print('viterbiProbTable', viterbiProbTable)
                        #print('k if(j==1)',k)
                        viterbiBackBackTable[:,j]=k
                        #print('viterbiBackBackTable', j, k, viterbiBackBackTable)
                        
                    else:
                        for x in range(rows):
                            # print(colProb)
                            # print('viterbiProbTable[x][j]', viterbiProbTable[x][j])
                            # print('colProb', colProb[x])
                            # print('viterbiProbTable[k][j-1]', viterbiProbTable[k][j-1])
                            if(viterbiProbTable[x][j]<colProb[x]+viterbiProbTable[k][j-1]):
                                viterbiProbTable[x][j]=colProb[x]+viterbiProbTable[k][j-1]
                                #print('viterbiProbTable', viterbiProbTable)
                                #print('k else(j==1)',k)
                                viterbiBackBackTable[x][j]=k
                                #print('viterbiBackBackTable', j, k, viterbiBackBackTable)
            
            elif(j!=0 and j!= 1):
                #print('j', j)
                for k in range(rows):
                    # prev_label= label_embeddings[k]
                    # prev_prev_label= label_embeddings[k]
                    
                    prev_label= label_embeddings[valid_y[i][j-1]]
                    prev_prev_label = label_embeddings[valid_y[i][j-2]]
                    #print("prev_label",j,prev_label)
                    #print("prev_prev_label",j,prev_prev_label)
                    #word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                    word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-2]],word_embeddings[valid_lex[i][j-1]],prev_prev_label,prev_label),1)
                    data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                    prediction = F.log_softmax(net(data), dim =1)
                    colProb=prediction.data.view(4).numpy()
                    # print(colProb)
                    # print(colProb[1])
                    if k==0:
                        viterbiProbTable[:,j]=colProb+viterbiProbTable[k][j-1]+viterbiProbTable[k][j-2]
                        #print('viterbiProbTable', viterbiProbTable)
                        #print('k 2nd if(j==1)',k)
                        viterbiBackBackTable[:,j]=k
                        #print('viterbiBackBackTable', j, k, viterbiBackBackTable)
                        #sys.exit()
                    else:
                        for x in range(rows):
                            # print(colProb)
                            # print(colProb[x])
                            if(viterbiProbTable[x][j]<colProb[x]+viterbiProbTable[k][j-1])+viterbiProbTable[k][j-2]:
                                viterbiProbTable[x][j]=colProb[x]+viterbiProbTable[k][j-1]+viterbiProbTable[k][j-2]
                                #print('viterbiProbTable', viterbiProbTable)
                                #print('k 2nd else',k)
                                viterbiBackBackTable[x][j]=k
                                #print('viterbiBackBackTable', j, k, viterbiBackBackTable)
                                #sys.exit()
        #print (viterbiProbTable)
        #print ("last",viterbiBackBackTable)
        #sys.exit()
        #BACKWARD PASS TO CREATE PATH
        output=[]
        for j in range(cols-1,-1,-1):
            if j==cols-1:
                #print(viterbiProbTable[:,j])
                row_index = viterbiProbTable[:,j].argmax(axis = 0)
                #print(row_index)
                output.append(row_index)
                prevLabel=viterbiBackBackTable[row_index][j]
                #print ("1st if",j, prevLabel)
            else:
                output.append(int(prevLabel))
                #print ("2nd if", j, prevLabel)
                # print viterbiBackBackTable[int(prevLabel)][j]
                prevLabel=viterbiBackBackTable[int(prevLabel)][j]
        output.reverse()
        #print (output,valid_y[i], len(output), len(valid_y[i]))
        #predictions_test = map(lambda t: idx2label[t], output)
        #print (predictions_test)
        #totalOutput.append(predictions_test)
        totalOutput.append(output)
        #sys.exit()
    #print("totalOutput", totalOutput, len(totalOutput))# 235
    return totalOutput

def evaluation(groundtruth_val, prediction):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    no_use = 0

    for i in range (len(groundtruth_val)):
        #print("i",len(groundtruth_val[i]))
        # print("groundtruth_val", groundtruth_val[i])
        # print("prediction", prediction[i])
        for j in range (len(groundtruth_val[i])):
            #print ('j',len(groundtruth_val[i][j]))
            if (groundtruth_val[i][j] == 'O' and prediction[i][j] == 'O'):
                no_use += 1

            if (groundtruth_val[i][j] == 'O' and prediction[i][j] == 'T-POS'):
                FP += 1
            if (groundtruth_val[i][j] == 'O' and prediction[i][j] == 'T-NEG'):
                #FN += 1
                FP += 1
            if (groundtruth_val[i][j] == 'O' and prediction[i][j] == 'T-NEU'):
                #FN += 1
                FP += 1

            if (groundtruth_val[i][j] == 'T-POS' and prediction[i][j] == 'O'):
                FN += 1
            if (groundtruth_val[i][j] == 'T-POS' and prediction[i][j] == 'T-POS'):
                TP += 1
            if (groundtruth_val[i][j] == 'T-POS' and prediction[i][j] == 'T-NEG'):
                FN += 1
                FP += 1 #new
            if (groundtruth_val[i][j] == 'T-POS' and prediction[i][j] == 'T-NEU'):
                FN += 1
                FP += 1

            if (groundtruth_val[i][j] == 'T-NEG' and prediction[i][j] == 'O'):
                FN += 1
            if (groundtruth_val[i][j] == 'T-NEG' and prediction[i][j] == 'T-POS'):
                FP += 1
                FN += 1 #new
            if (groundtruth_val[i][j] == 'T-NEG' and prediction[i][j] == 'T-NEG'):
                #TN += 1
                TP += 1 #new
            if (groundtruth_val[i][j] == 'T-NEG' and prediction[i][j] == 'T-NEU'):
                FN += 1
                FP += 1

            if (groundtruth_val[i][j] == 'T-NEU' and prediction[i][j] == 'O'):
                FN += 1
            if (groundtruth_val[i][j] == 'T-NEU' and prediction[i][j] == 'T-POS'):
                FP += 1
                FN += 1 #new
            if (groundtruth_val[i][j] == 'T-NEU' and prediction[i][j] == 'T-NEG'):
                FN += 1
                FP += 1 #new
            if (groundtruth_val[i][j] == 'T-NEU' and prediction[i][j] == 'T-NEU'):
                TP += 1
                #TN += 1
    # print('no_use', no_use) 
    # print('TP', TP) 
    # print('TN', TN) 
    # print('FP', FP) 
    # print('FN', FN) 

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    f1 =  (2 * precision * recall) / (precision + recall)

    count_acc = 0
    for i in range (len(groundtruth_val)):
        for j in range (len(groundtruth_val[i])): 
            #if (groundtruth_val[i][j] != 'O' and prediction[i][j] != 'O'):
            if (groundtruth_val[i][j] != 'O' ):

                if (groundtruth_val[i][j] == prediction[i][j]):
                    count_acc = count_acc + 1
        
    #print("count_acc", count_acc) #3255

    total_element = 0
    # for i in groundtruth_val:
    #     if (groundtruth_val[i][j] != 'O' and prediction[i][j] != 'O'):
    #         total_element += len(i)    
    for i in range (len(groundtruth_val)):
        for j in range (len(groundtruth_val[i])): 
            #if (groundtruth_val[i][j] != 'O' and prediction[i][j] != 'O'):
            if (groundtruth_val[i][j] != 'O' ):
                total_element = total_element +1
    #print('total_element', total_element) #3926  

    accuracy =  count_acc /  total_element       

    # print("Precision: ", precision) #0.069
    # print("Recall: ", recall) #0.0047
    # print("F1: ", f1) #0.008
    # print("accuracy: ", accuracy) # 0.829

    return precision, recall, f1, accuracy 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    #parser.add_argument('--valid_file', type=str, default='data/twitter1_valid.txt', help='Valid file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    args = parser.parse_args()

    # read the dataset
    train_set = read_data(path=args.train_file)
    test_set = read_data(path=args.test_file)
    #valid_set = read_data(path=args.valid_file)

    # uncomment if you want to see the data format
    #print(train_set[0], type(train_set)) ##list type
    # print(train_set[0]['sentence']) #How can someone so incompetent like Maxine Waters stay in office for over 20 years ? #LAFail
    # print(train_set[0]['words']) #['how', 'can', 'someone', 'so', 'incompetent', 'like', 'maxine', 'waters', 'stay', 'in', 'office', 'for', 'over', '20', 'years', 'PUNCT', '#lafail']
    # print(train_set[0]['ts_raw_tags']) #['O', 'O', 'O', 'O', 'O', 'O', 'T-NEG', 'T-NEG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    

    # now, you must parse the dataset

    if (args.option == 1): #######  Randomly Initialized
        
        tag_list = ['T-POS', 'T-NEG', 'T-NEU', 'O']
        labeled_features = []

        #dimensionsOrg = 300
        dimensionsOrg = 10
        #****************************************************************building input features part.1
        word_to_ix = build_index_vocab(train_set)
        #print("train", word_to_ix,len(word_to_ix)) #10850
        tag_to_ix = {"O": 0, "T-POS": 1, "T-NEG": 2, "T-NEU": 3, 'START': 4}
        #print(tag_to_ix)

        word_embeddings = create_word_embeddings(word_to_ix,dimensionsOrg)
        label_embeddings = create_label_embeddings(tag_to_ix,dimensionsOrg)

        #print (word_embeddings, word_embeddings[0], word_embeddings[0][0])
        #print (label_embeddings)
        #sys.exit()

        sentence_in =[]
        targets =[]
        for i in range(len(train_set)):
            sentence_in.append(prepare_sequence(train_set[i]['words'], word_to_ix))
            targets.append(prepare_sequence(train_set[i]['ts_raw_tags'], tag_to_ix))
           
        # print(sentence_in, len(sentence_in)) #2115
        # print(targets, len(targets)) #2115

        ###build feature and start training####
        
        startTraining(word_embeddings,label_embeddings,sentence_in, targets, dimensionsOrg)
        #sys.exit()

        # groundtruth_val = []
        # for i in targets:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     groundtruth_val.append(temp)
        # #print (groundtruth_val, len(groundtruth_val), type(groundtruth_val)) #235 <class 'list'>
        # #print (groundtruth_val)
        
        # ####### .......predict with viterbi.......######
        # preds_viterbi=viterbi(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        # ### conceptual question #####
        # ##.......predict with Neural Net.......###
        # preds_NN = predict_with_neural_net(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        # #print('predictions_val', predictions_val, type(predictions_val)) #<class 'list'>
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        # prediction_viterbi = []
        # for i in preds_viterbi:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_viterbi.append(temp)
        # ########### Evaluation Calculation Train set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Train Performance option 1....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        # print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )

        # ########### Validation set ###############

        # word_to_ix = build_index_vocab(valid_set)
        # #print("test ", word_to_ix,len(word_to_ix)) #1922
        # word_embeddings = create_word_embeddings(word_to_ix,dimensionsOrg)
        # label_embeddings = create_label_embeddings(tag_to_ix,dimensionsOrg)

        # # print ("test wordem",word_embeddings, word_embeddings[0], word_embeddings[0][0])
        # # print ("test labelem", label_embeddings)

        # sentence_in =[]
        # targets =[]
        # for i in range(len(valid_set)):
        #     sentence_in.append(prepare_sequence(valid_set[i]['words'], word_to_ix))
        #     #print('ba', prepare_sequence(valid_set[i]['ts_raw_tags'], tag_to_ix))
        #     targets.append(prepare_sequence(valid_set[i]['ts_raw_tags'], tag_to_ix))
        # groundtruth_val = []
        # for i in targets:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     groundtruth_val.append(temp)
        # #print (groundtruth_val, len(groundtruth_val), type(groundtruth_val)) #235 <class 'list'>
        # #print (groundtruth_val)
        
        # ####### .......predict with viterbi.......######
        # preds_viterbi=viterbi(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        # ### conceptual question #####
        # ##.......predict with Neural Net.......###
        # preds_NN = predict_with_neural_net(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        # #print('predictions_val', predictions_val, type(predictions_val)) #<class 'list'>
        # ###predict with random

        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        # prediction_viterbi = []
        # for i in preds_viterbi:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_viterbi.append(temp)
        # ########### Evaluation Calculation Validation set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Validation Performance option 1....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        # print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )
        
        ############ Test Set ###############
        #print ("hi test")
        
        word_to_ix = build_index_vocab(test_set)
        #print("test ", word_to_ix,len(word_to_ix)) #1922
        word_embeddings = create_word_embeddings(word_to_ix,dimensionsOrg)
        label_embeddings = create_label_embeddings(tag_to_ix,dimensionsOrg)

        # print ("test wordem",word_embeddings, word_embeddings[0], word_embeddings[0][0])
        # print ("test labelem", label_embeddings)

        sentence_in =[]
        targets =[]
        for i in range(len(test_set)):
            sentence_in.append(prepare_sequence(test_set[i]['words'], word_to_ix))
            #print('ba', prepare_sequence(test_set[i]['ts_raw_tags'], tag_to_ix))
            targets.append(prepare_sequence(test_set[i]['ts_raw_tags'], tag_to_ix))
        groundtruth_val = []
        for i in targets:
            #print (i, type(i))
            temp = []
            for j in i:
                #print (j)
                #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
                temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
            #print(temp)
            groundtruth_val.append(temp)
        #print (groundtruth_val, len(groundtruth_val), type(groundtruth_val)) #235 <class 'list'>
        #print (groundtruth_val)
        
        ####### predict with viterbi######
        preds_viterbi=viterbi(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        ### conceptual question #####
        ##.......predict with Neural Net.......###
        #preds_NN = predict_with_neural_net(dimensionsOrg, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
       

        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        prediction_viterbi = []
        for i in preds_viterbi:
            #print (i, type(i))
            temp = []
            for j in i:
                #print (j)
                #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
                temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
            #print(temp)
            prediction_viterbi.append(temp)
        ########### Evaluation Calculation Test set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Test Performance option 1....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        #print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )
        print("Precision: ", precision) 
        print("Recall: ", recall) 
        print("F1: ", f1) 
        # print("accuracy: ", accuracy) 


    if (args.option == 2):  ######  Word2Vec
        # example to load the Word2Vec model
        # note: this will only work on a cs machine (ex: data.cs.purdue.edu)
        wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
        # you can get the vector for a word like so
        #vector = wv_from_bin['man']
        #print (vector)
        #sys.exit()
        #print(vector.shape) #(300,)

        # weights to use in from_pretrained()
        weights = torch.FloatTensor(wv_from_bin.vectors)
        #print (weights, weights.size()) #torch.Size([3000000, 300])

        #print (weights[0], weights[0].size()) # torch.Size([300])
        
        dimensionsOrg = 300
        tag_dim = 4
        tag_to_ix = {"O": 0, "T-POS": 1, "T-NEG": 2, "T-NEU": 3, 'START': 4}
        # #print(tag_to_ix)

        ############## Train Set ################
        #### build vocabulary according to word embedding
        word_to_ix = build_vocab_using_word_embedding(train_set, wv_from_bin)
        # Adding a new key value pair for unknown word
        word_to_ix.update( {'my_unknown' : len(word_to_ix)} )  
        word_embeddings = create_word_embeddings_w2v(weights)
        label_embeddings = create_label_embeddings(tag_to_ix,tag_dim)

        #print (word_embeddings, word_embeddings[0])
        sentence_in =[]
        targets =[]
        for i in range(len(train_set)):
            #sentence_in.append(prepare_sequence_word_em(train_set[i]['words'], word_to_ix))
            #targets.append(prepare_sequence_word_em(train_set[i]['ts_raw_tags'], tag_to_ix))
            targets.append(prepare_sequence(train_set[i]['ts_raw_tags'], tag_to_ix))
            #print (targets[0], len(targets[0])) #17
           
            sequence = prepare_sequence_word_em(train_set[i]['words'], word_to_ix)
            embeds = word_embeddings(sequence)
            sentence_in.append(embeds)
            

        ###build feature and start training####
        
        startTraining_option2(tag_to_ix,label_embeddings,sentence_in, targets, dimensionsOrg, tag_dim)

        # groundtruth_val = []
        # for i in targets:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     groundtruth_val.append(temp)
        # ###....predict with viterbi
        # preds_viterbi=viterbi_option2(dimensionsOrg, tag_dim,  tag_to_ix, label_embeddings, sentence_in, targets)

        # ### conceptual question #####
        # ##.......predict with Neural Net.......###
        # preds_NN = predict_with_neural_net_option2(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        # prediction_viterbi = []
        # for i in preds_viterbi:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_viterbi.append(temp)
        # ########### Evaluation Calculation Train set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Train Performance option 2....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        # print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )


        # ############ Validation Set ###############
        # #print ("hi test")
        
        # #### build vocabulary according to word embedding
        # word_to_ix = build_vocab_using_word_embedding(valid_set, wv_from_bin)
        # # Adding a new key value pair for unknown word
        # word_to_ix.update( {'my_unknown' : len(word_to_ix)} )  
        # #print("train", word_to_ix,len(word_to_ix)) # 6413


        # # tag_to_ix = build_vocab_tag_using_word_embedding(test_set, wv_from_bin)
        # # print("train tag ", tag_to_ix,len(tag_to_ix)) 


        # #word_embeddings = create_word_embeddings_w2v(word_to_ix,dimensionsOrg)
        # word_embeddings = create_word_embeddings_w2v(weights)
        # label_embeddings = create_label_embeddings(tag_to_ix,tag_dim)

        # #print (word_embeddings, word_embeddings[0])
        # sentence_in =[]
        # targets =[]
        # for i in range(len(valid_set)):
        #     targets.append(prepare_sequence(valid_set[i]['ts_raw_tags'], tag_to_ix))
        #     sequence = prepare_sequence_word_em(valid_set[i]['words'], word_to_ix)
        #     embeds = word_embeddings(sequence)
        #     sentence_in.append(embeds)
            
        # groundtruth_val = []
        # for i in targets:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     groundtruth_val.append(temp)
        # ###....predict with viterbi
        # preds_viterbi=viterbi_option2(dimensionsOrg, tag_dim,  tag_to_ix, label_embeddings, sentence_in, targets)

        # ### conceptual question #####
        # ##.......predict with Neural Net.......###
        # preds_NN = predict_with_neural_net_option2(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        # prediction_viterbi = []
        # for i in preds_viterbi:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_viterbi.append(temp)
        # ########### Evaluation Calculation Validation set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Validation Performance option 2....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        # print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )


        ############ Test Set ###############
        #print ("hi test")
        
        #### build vocabulary according to word embedding
        word_to_ix = build_vocab_using_word_embedding(test_set, wv_from_bin)
        # Adding a new key value pair for unknown word
        word_to_ix.update( {'my_unknown' : len(word_to_ix)} )  
        #print("train", word_to_ix,len(word_to_ix)) # 6413


        # tag_to_ix = build_vocab_tag_using_word_embedding(test_set, wv_from_bin)
        # print("train tag ", tag_to_ix,len(tag_to_ix)) 


        #word_embeddings = create_word_embeddings_w2v(word_to_ix,dimensionsOrg)
        word_embeddings = create_word_embeddings_w2v(weights)
        label_embeddings = create_label_embeddings(tag_to_ix,tag_dim)

        #print (word_embeddings, word_embeddings[0])
        sentence_in =[]
        targets =[]
        for i in range(len(test_set)):
            targets.append(prepare_sequence(test_set[i]['ts_raw_tags'], tag_to_ix))
            sequence = prepare_sequence_word_em(test_set[i]['words'], word_to_ix)
            embeds = word_embeddings(sequence)
            sentence_in.append(embeds)
            
        groundtruth_val = []
        for i in targets:
            #print (i, type(i))
            temp = []
            for j in i:
                #print (j)
                #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
                temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
            #print(temp)
            groundtruth_val.append(temp)
        ###....predict with viterbi#####
        preds_viterbi=viterbi_option2(dimensionsOrg, tag_dim, tag_to_ix, label_embeddings, sentence_in, targets)

        ### conceptual question #####
        ##.......predict with Neural Net.......###
        #preds_NN = predict_with_neural_net_option2(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        prediction_viterbi = []
        for i in preds_viterbi:
            #print (i, type(i))
            temp = []
            for j in i:
                #print (j)
                #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
                temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
            #print(temp)
            prediction_viterbi.append(temp)
        ########### Evaluation Calculation Test set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Test Performance option 2....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        #print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )
        print("Precision: ", precision) 
        print("Recall: ", recall) 
        print("F1: ", f1) 
            
    if (args.option == 3):  ######  Bi-lstm
        # example to load the Word2Vec model
        # note: this will only work on a cs machine (ex: data.cs.purdue.edu)
        wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
        
        # you can get the vector for a word like so
        #vector = wv_from_bin['man']
        #print (vector)
        #sys.exit()
        #print(vector.shape) #(300,)

        # weights to use in from_pretrained()
        weights = torch.FloatTensor(wv_from_bin.vectors)
        #print (weights, weights.size()) #torch.Size([3000000, 300])

        #print (weights[0], weights[0].size()) # torch.Size([300])
        
        dimensionsOrg = 300
        tag_dim = 4
        tag_to_ix = {"O": 0, "T-POS": 1, "T-NEG": 2, "T-NEU": 3, 'START': 4}
        # #print(tag_to_ix)

        ############## Train Set ################
        #### build vocabulary according to word embedding
        word_to_ix = build_vocab_using_word_embedding(train_set, wv_from_bin)
        # Adding a new key value pair for unknown word
        word_to_ix.update( {'my_unknown' : len(word_to_ix)} )  
        word_embeddings = create_word_embeddings_w2v(weights)
        label_embeddings = create_label_embeddings(tag_to_ix,tag_dim)

        #print (word_embeddings, word_embeddings[0])
        sentence_in =[]
        targets =[]
        for i in range(len(train_set)):
            #sentence_in.append(prepare_sequence_word_em(train_set[i]['words'], word_to_ix))
            #targets.append(prepare_sequence_word_em(train_set[i]['ts_raw_tags'], tag_to_ix))
            targets.append(prepare_sequence(train_set[i]['ts_raw_tags'], tag_to_ix))
            #print (targets[0], len(targets[0])) #17
           
            sequence = prepare_sequence_word_em(train_set[i]['words'], word_to_ix)
            embeds = word_embeddings(sequence)
            sentence_in.append(embeds)
            

        ###build feature and start training####
        
        startTraining_option3(tag_to_ix,label_embeddings,sentence_in, targets, dimensionsOrg, tag_dim)

        # groundtruth_val = []
        # for i in targets:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     groundtruth_val.append(temp)
        # ###....predict with viterbi
        # preds_viterbi=viterbi_option3(dimensionsOrg, tag_dim,  tag_to_ix, label_embeddings, sentence_in, targets)

        # ### conceptual question #####
        # ##.......predict with Neural Net.......###
        # preds_NN = predict_with_neural_net_option3(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        # prediction_viterbi = []
        # for i in preds_viterbi:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_viterbi.append(temp)
        # ########### Evaluation Calculation Train set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Train Performance option 3....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        # print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )

        # ############ Validation Set ###############
        # #print ("hi test")
        
        # #### build vocabulary according to word embedding
        # word_to_ix = build_vocab_using_word_embedding(valid_set, wv_from_bin)
        # # Adding a new key value pair for unknown word
        # word_to_ix.update( {'my_unknown' : len(word_to_ix)} )  
        # #print("train", word_to_ix,len(word_to_ix)) # 6413


        # # tag_to_ix = build_vocab_tag_using_word_embedding(test_set, wv_from_bin)
        # # print("train tag ", tag_to_ix,len(tag_to_ix)) 


        # #word_embeddings = create_word_embeddings_w2v(word_to_ix,dimensionsOrg)
        # word_embeddings = create_word_embeddings_w2v(weights)
        # label_embeddings = create_label_embeddings(tag_to_ix,tag_dim)

        # #print (word_embeddings, word_embeddings[0])
        # sentence_in =[]
        # targets =[]
        # for i in range(len(valid_set)):
        #     targets.append(prepare_sequence(valid_set[i]['ts_raw_tags'], tag_to_ix))
        #     sequence = prepare_sequence_word_em(valid_set[i]['words'], word_to_ix)
        #     embeds = word_embeddings(sequence)
        #     sentence_in.append(embeds)
            
        # groundtruth_val = []
        # for i in targets:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     groundtruth_val.append(temp)
        # ###....predict with viterbi
        # preds_viterbi=viterbi_option3(dimensionsOrg, tag_dim,  tag_to_ix, label_embeddings, sentence_in, targets)

        # ### conceptual question #####
        # ##.......predict with Neural Net.......###
        # preds_NN = predict_with_neural_net_option3(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        # prediction_viterbi = []
        # for i in preds_viterbi:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_viterbi.append(temp)
        # ########### Evaluation Calculation Validation set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Validation Performance option 3....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        # print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )


        ############ Test Set ###############
        #print ("hi test")
        
        #### build vocabulary according to word embedding
        word_to_ix = build_vocab_using_word_embedding(test_set, wv_from_bin)
        # Adding a new key value pair for unknown word
        word_to_ix.update( {'my_unknown' : len(word_to_ix)} )  
        #print("train", word_to_ix,len(word_to_ix)) # 6413


        # tag_to_ix = build_vocab_tag_using_word_embedding(test_set, wv_from_bin)
        # print("train tag ", tag_to_ix,len(tag_to_ix)) 


        #word_embeddings = create_word_embeddings_w2v(word_to_ix,dimensionsOrg)
        word_embeddings = create_word_embeddings_w2v(weights)
        label_embeddings = create_label_embeddings(tag_to_ix,tag_dim)

        #print (word_embeddings, word_embeddings[0])
        sentence_in =[]
        targets =[]
        for i in range(len(test_set)):
            targets.append(prepare_sequence(test_set[i]['ts_raw_tags'], tag_to_ix))
            sequence = prepare_sequence_word_em(test_set[i]['words'], word_to_ix)
            embeds = word_embeddings(sequence)
            sentence_in.append(embeds)
            
        groundtruth_val = []
        for i in targets:
            #print (i, type(i))
            temp = []
            for j in i:
                #print (j)
                #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
                temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
            #print(temp)
            groundtruth_val.append(temp)
        ###....predict with viterbi#####
        preds_viterbi=viterbi_option3(dimensionsOrg, tag_dim, tag_to_ix, label_embeddings, sentence_in, targets)

        ### conceptual question #####
        ##.......predict with Neural Net.......###
        #preds_NN = predict_with_neural_net_option3(dimensionsOrg, tag_dim, word_to_ix, tag_to_ix, word_embeddings, label_embeddings, sentence_in, targets)
        
        # prediction_NN = []
        # for i in preds_NN:
        #     #print (i, type(i))
        #     temp = []
        #     for j in i:
        #         #print (j)
        #         #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #         temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
        #     #print(temp)
        #     prediction_NN.append(temp)

        prediction_viterbi = []
        for i in preds_viterbi:
            #print (i, type(i))
            temp = []
            for j in i:
                #print (j)
                #print(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
                temp.append(list(tag_to_ix.keys())[list(tag_to_ix.values()).index(j)])
            #print(temp)
            prediction_viterbi.append(temp)
        ########### Evaluation Calculation Test set##########
        # precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_NN)
        # print(".....Test Performance option 3....")
        # print('precision, recall, f1, accuracy NN: ', precision, recall, f1, accuracy )
        precision, recall, f1, accuracy = evaluation(groundtruth_val, prediction_viterbi)
        #print('precision, recall, f1, accuracy Viterbi: ', precision, recall, f1, accuracy )
        print("Precision: ", precision) 
        print("Recall: ", recall) 
        print("F1: ", f1) 
 

if __name__ == '__main__':
    main()   
