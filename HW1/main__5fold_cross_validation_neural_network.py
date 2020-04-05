import csv
from random import randrange
import pandas as pd
import numpy as np


###my change
import math
import re
import sys
from nltk.stem import WordNetLemmatizer
from string import punctuation as punc
import spacy  # For preprocessing
import nltk
import string
from nltk.corpus import stopwords
import preprocessor as p  #pip install tweet-preprocessor
import logging  # Setting up the loggings to monitor gensim
import gensim
from gensim.models.phrases import Phrases, Phraser
nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed #python -m spacy download en
import multiprocessing
from gensim.models import Word2Vec
import time
import torch
import minibatcher
import statistics 
import pprint
from collections import Counter
import heapq
from numpy.linalg import norm
import scipy.sparse
import random
from numpy import linalg as LNG 

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore') 

learning_rate = 0.5
_lambda = 0.1
no_class = 17
EPSILON = 1e-14
np.random.seed(1) ## seed

def init_weights(embedding_dim):
    #model = dict ( W1 = np.random.randn(embedding_dim + 1, no_class+1)) ## adding +1 for bias dimension, +1 for label 1hot
    #print (model["W1"], model["W1"].shape) #(2501, 18)
    #mu, sigma = 0, 0.0001
    mu, sigma = 0, 0.1
    model = dict ( W1 = np.random.normal(mu, sigma, [embedding_dim + 1, no_class+1]))
    #model = dict ( W1 =np.zeros([embedding_dim + 1, no_class + 1])) ## adding +1 for bias dimension, +1 for label 1hot
    #model = dict ( W1 =np.zeros([tf_idf_model.shape[1] ,len(np.unique(y_train)) + 1]))
    #print ("initial ", model["W1"])
    return model

def softmax(x):
    #return np.exp(x) / np.exp(x).sum()
    z = x - np.max(x)
    return np.exp(z) / np.exp(z).sum()

def stable_softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

# My Change.  Define Sigmoid function. 
def sigmoid(z_1):
    sigmoid_scores = [1 / float(1 + np.exp(- x)) for x in z_1]
    return sigmoid_scores

def oneHotIt(Y):
    m = Y.shape[0]
    y_hot = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    y_hot = np.array(y_hot.todense()).T
    return y_hot
def forward(x, model):
    #print (x.shape) # #(1230, 2500)
    #x = np.append(x, 1) #1 for bias problem
    #print (x.shape)
    #print(model['W1'].shape) #(2501, 18)
    x = np.hstack([x, np.ones([x.shape[0], 1])]) #bias is in last column
    #x = np.hstack([np.ones([x.shape[0],1]), x])  #bias is in first column
    z1 = np.dot(x , model['W1'])  #input times first layer matrix
    #print (z1.shape)  #(1230, 17)
    hat_y = stable_softmax(z1)  # output layer activation
    #print ("hat_y", hat_y, len(hat_y)) #1 * 17
    return hat_y

def backward(xs, errs, model):
    #dw1 =  xs.T @ errs
    #print ("dw1", dw1)
    #print ("xs.shape[0]", xs.shape[0]) # 1230
    
    # Add the 1 to the data, to compute the gradient of W1
    #xs = np.hstack([xs, np.ones([xs.shape[0], 1])])

    # The bias "neuron" is the constant 1, we don't need to backpropagate its gradient
    # since it has no inputs, so we just make 0 of its column from the gradient
    xs = np.hstack([xs, np.zeros([xs.shape[0], 1])])
    dw1 = (xs.T @ errs)/ xs.shape[0]  # errs is the gradient of output layer
    dw1 = dw1 + (_lambda * model['W1']) ##add regularization
    #print ("dw1.shape",dw1.shape) # (2501, 18)

    #print ("dw1", dw1)
    return dict(W1 = dw1)

def get_gradient(X_train, y_train, model):
    #print (X_train.shape) #(1230, 2500)
    y_pred = forward(X_train, model)
    #print ("y_pred", y_pred)  #1 * 17
    #create one hot encoding of true label
    
    #y_true = np.zeros(no_class+1)
    #y_true[int(cls_indx)] = 1

    y_true = oneHotIt (y_train)

    #print(y_true)

    #compute gradient of output layer
    err = y_true - y_pred
    return backward(X_train, err, model)

def gradient_step(X_train, y_train, model): #define single gradient ascent step here we update parameters
    #print (X_train.shape) #(1230, 2500)
    grad = get_gradient (X_train, y_train, model) 

    #update every parameter of network using their gradients
    for param in grad:
        #print (param)
        model[param] += learning_rate * grad[param]
    #print ("after update", model["W1"])
    return model
'''
def gradient_ascent(X_train, y_train, model):  #repeat gradient ascent for few more times based on epochs
    for i in range(n_epochs):
        print('Iteration {}'.format(i))
        #print (X_train.shape) #(1230, 2500)
        model = gradient_step(X_train, y_train, model)
    return model
'''
def find_accuracy(tf_idf_model, y_train, model):
    probs = forward(tf_idf_model, model)
    #print(probs, probs.shape) ###(1230, 18)
    preds = np.argmax(probs,axis=1)
    #print (preds)
    count = 0
    for i in range (0, len(preds)):
        if (preds[i] == y_train[i]):
            count += 1
    print ("count", count)
    accuracy = sum(preds == y_train)/(float(len(y_train)))
    return accuracy

def get_test_label(tf_idf_model, model):
    probs = forward(tf_idf_model, model)
    #print(probs, probs.shape) ###(1230, 18)
    preds = np.argmax(probs,axis=1)
    return preds

def _l2_loss(model):
    l2_reg = None
    #print (model.values())
    for w in model.values():
        if l2_reg is None:
            l2_reg = LNG.norm(w)
        else:
            l2_reg += LNG.norm(w)
    return l2_reg

def find_loss(X_train, y_train, model):
    y_true = oneHotIt (y_train)
    prob = forward(X_train, model)
    #loss = (-1 /  tf_idf_model.shape[0]) * np.sum(y_true * np.log(probs)) ####without regularization
    #print ("loss", loss)
    loss = np.sum((-1 /  X_train.shape[0]) * np.sum(y_true * np.log(prob + EPSILON))) + ((_lambda/2)* _l2_loss(model))  ###with regularization
    print ("nll loss", loss)
    return loss

def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label

    #print("Read", row_count, "data points...")
    return tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label


def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):

    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])

def remove_string_noise(input_str):
    input_str = re.sub(r"http\S+", "", input_str)
    
    #give special char you want to remove
    #do not put space between chars, and space (" ") is not a special char
    punctuation_noise ="!\"$%&'#()*+,-./:;<=>?@[\]^_`{|}~" #print string.punctuation 
    number_noise = "0123456789"
    special_noise = ""

    all_noise = punctuation_noise + number_noise + special_noise
    #all_noise = punctuation_noise + special_noise

    for c in all_noise:
        if c in input_str:
            input_str = input_str.replace(c, " ")#replace with space
    fresh_str = ' '.join(input_str.split())
    return fresh_str

def clean_tweets(tweet):
    #HappyEmoticons
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
        ])

    # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])

    #combine sad and happy emoticons
    emoticons = emoticons_happy.union(emoticons_sad)

    wordnet_lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words('english'))
    operators = set(('no', 'not', 'nor', 'none'))
    stop_words = set(sw) - operators
    stop_words.update([ 'amp', 'rt'])  ###as we are using set so we used .update....otherwise .extends
    
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    word_tokens = nltk.word_tokenize(tweet)
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words and w not in emoticons:
        #if w not in emoticons:
            filtered_tweet.append(w)
    lemmatized_tweet = []
    for word in filtered_tweet:
        lemmatized_tweet.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    #print (sentences)
    #return ' '.join(filtered_tweet)
    return ' '.join(lemmatized_tweet)
    #return ' '.join(sentences)
    
    
def pre_processing_tweets (df):
    #p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.RESERVED)
    p.set_options( p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.RESERVED)
    
    clean_text = []
    #for i in range (0, df.shape[0]):
        #clean_text.append(p.clean(str(df['text'][i])))#python 3

    for i in df['text']:
        clean_text.append(p.clean(str(i)))#python 3

    fresh_text1 = []
    for i in range (0, df.shape[0]):
        fresh_text1.append(remove_string_noise(clean_text[i].encode('ascii', 'ignore').decode("utf-8"))) #can remove other emojis and no \UF..

    #Call clean_tweet method-2 for extra preprocessing
    filtered_tweet = []
    for i in range (0, len(fresh_text1)):
        filtered_tweet.append(clean_tweets(fresh_text1[i].lower()))

    #####for bigram
    
    df_clean = pd.DataFrame({'clean': filtered_tweet})
    #df_clean = df_clean.dropna().drop_duplicates()
    sent = [row.split() for row in df_clean['clean']]
    #print (sent_word, type(sent_word))

    ## create bigram
    #Creates the relevant phrases from the list of sentences:
    phrases = Phrases(sent, min_count=1, progress_per=10000)
    #print (phrases, type(phrases))

    #The goal of Phraser() is to cut down memory consumption of Phrases(), by discarding model state not strictly needed for the bigram detection task:
    bigram = Phraser(phrases)
    #print (bigram)

    #Transform the corpus based on the bigrams detected:
    sentences = bigram[sent]
    training_data  = [sentences[i] for i in range(0,len(sentences))]
    
    #return filtered_tweet
    return training_data

def build_vocabulary (training_data):
    wordfreq = {}
    for tokens in training_data:
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    #print (wordfreq, len(wordfreq))  #unigram = 2814 , bigram = 3499, trigram = 3499

    vocab = list(wordfreq.keys()) #convert dictionary keys to list
    return vocab

def build_IDF (training_data, vocab):
     ###IDF
    word_idf_values = {}
    for token in vocab:
        doc_containing_word = 0
        for tokens in training_data:
            if token in tokens:
                doc_containing_word += 1
        word_idf_values[token] = np.log(len(training_data)/(1 + doc_containing_word))
    return word_idf_values

def build_representation (training_data, vocab, word_idf_values):
    ####TF-IDF approach

    #print("word_idf_values", word_idf_values, len(word_idf_values)) #len(word_freq)

    ## TF
    word_tf_values = {}
    for token in vocab:
        sent_tf_vector = []
        for tokens in training_data:
            doc_freq = 0
            for word in tokens:
                if token == word:
                    doc_freq += 1
            word_tf = doc_freq/len(tokens)
            sent_tf_vector.append(word_tf)
        word_tf_values[token] = sent_tf_vector
    #print("word_tf_values", word_tf_values, len(word_tf_values)) #len(word_freq)

    tfidf_values = []
    for token in word_tf_values.keys():
        tfidf_sentences = []
        for tf_sentence in word_tf_values[token]:
            tf_idf_score = tf_sentence * word_idf_values[token]
            tfidf_sentences.append(tf_idf_score)
        tfidf_values.append(tfidf_sentences)
    
    tf_idf_model = np.asarray(tfidf_values)

    #print ("tf_idf_model", tf_idf_model, tf_idf_model.shape)  #len(word_freq) * 1230

    tf_idf_model = tf_idf_model.T

    
    ################ Normalization of data
    #print ("tf_idf_model before normalization", tf_idf_model, len(tf_idf_model)) # 1230 * embedding_dim
    #print ("min, max before normalization", np.min(tf_idf_model), np.max(tf_idf_model))
    #tf_idf_model = 1 * ((tf_idf_model - np.min(tf_idf_model)) / ( np.max(tf_idf_model) - np.min(tf_idf_model)))
    
    return tf_idf_model

def data_normalization(data, minimum, maximum):
    data = 1 * ((data - minimum) / ( maximum - minimum))
    
    return data


def save_plots_accs( train_accs, test_accs):
    n = len(train_accs)
    xs = np.arange(n)
    
    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='validation')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy.png')


def LR():

    # Read training data as dictionary
    #train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')
    df = pd.read_csv("train.csv")
    
    y_train = df.label.values 

    ##########.....Train data pre-processing....... ######
    training_data = pre_processing_tweets(df)
    #print("training_data", type(training_data))
    ##########.....Build vocabulary on preprpcessed Train data....... ######
    vocab = build_vocabulary(training_data)

    ##########.....Build IDF on preprpcessed Train data....... ######
    word_idf_values = build_IDF(training_data, vocab)

    ##### Build TF-DF representation of train data ########
    tf_idf_model = build_representation(training_data, vocab, word_idf_values)
    minimum = np.min(tf_idf_model)
    maximum = np.max(tf_idf_model)
    ########### Normalize train data #########
    tf_idf_model_train = data_normalization(tf_idf_model, minimum, maximum)

    
    model = init_weights(len(vocab))

    for i in range(n_epochs):

        print('Iteration {}'.format(i))
        #print (X_train.shape) #(1230, 2500)
        model = gradient_step(tf_idf_model_train, y_train, model)
        

        ###find train accuracy
        #print ('Training Accuracy prev: ', find_accuracy(sentence_vectors, y_train, model )) ###using BOW
        acc = find_accuracy(tf_idf_model_train, y_train, model )
        #print ('Training Accuracy scratch: ', acc) ###using TF-IDF
        loss = find_loss(tf_idf_model_train, y_train, model )
        #print ('Training Loss scratch: ', loss) ###using TF-IDF 

    #save_plots_accs(accs, accs_val)   #ok

    # Read test data
    
    #test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    df_test = pd.read_csv("test.csv") 
    #print (df.shape) #820,5
    #print (df.head())
    test_data = pre_processing_tweets(df_test)
    tf_idf_model_test = build_representation(test_data, vocab, word_idf_values)

    #### data normalization
    ####replace the elements of np array which is greater than maximum with maximum
    #arr[arr > 255] = x
    tf_idf_model_test[tf_idf_model_test > maximum] = maximum
    ####replace the elements of np array which is smaller than minimum with minimum
    tf_idf_model_test[tf_idf_model_test < minimum] = minimum

    tf_idf_model_test = data_normalization(tf_idf_model_test, minimum, maximum)

    # Predict test data by learned model
    test_pred = get_test_label(tf_idf_model_test, model)
    #print (test_pred, len(test_pred)) #820
    

    
    #Replace the following random predictor by your prediction function.
    

    #df_test = pd.DataFrame({'label': test_pred})
    df_test_new = pd.DataFrame({'tweet_id': df_test.tweet_id})
    df_test_new['issue'] = df_test.issue
    df_test_new['text'] = df_test.text
    df_test_new['author'] = df_test.author
    df_test_new['label'] = test_pred
    #df_test_new.to_csv('test_lr.csv')
   

    # Save predicted labels in 'test_lr.csv'
    #SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_lr.csv')
    
################## Multi Layer Neural Network ##################

#learning_rate_nn = 1

learning_rate_nn = [.01,.05, .1, .5, 1]
_lambda_nn = 0.0001
# n_epochs = 462
# n_hidden=300
n_epochs=200
n_hidden = 500
############### 1 hidden layer MLP with ReLu ############

def init_weights_nn(embedding_dim):
    # # Initialize weights with Standard Normal random variables
    # model = dict(
    #     W1=np.random.randn(embedding_dim + 1, n_hidden),
    #     W2=np.random.randn(n_hidden + 1, no_class)
    # )

    mu, sigma = 0, 0.1
    model = dict ( W1 = np.random.normal(mu, sigma, [embedding_dim + 1, n_hidden]), 
                W2 = np.random.normal(mu, sigma, [n_hidden + 1, no_class]))
    return model


# For a single example $x$
def forward_nn(x, model):
    x = np.append(x, 1)
    
    # Input times first layer matrix 
    z_1 = x @ model['W1']

    # ReLU activation goes to hidden layer
    h = z_1
    h[z_1 < 0] = 0

    # Hidden layer values to output
    h = np.append(h, 1)
    hat_y = softmax(h @ model['W2'])

    return h, hat_y

def backward_nn(xs, hs, errs, model):
    #print (xs.shape)
    #print (hs.shape)
    #print (errs.shape)
    
    """xs, hs, errs contain all information (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    dW2 = (hs.T @ errs)/xs.shape[0]
    dW2 = dW2 + (_lambda_nn * model['W2']) # Add regularization

    # Get gradient of hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0
    
    # The bias "neuron" is the constant 1, we don't need to backpropagate its gradient
    # since it has no inputs, so we just remove its column from the gradient
    dh = dh[:, :-1]

    # Add the 1 to the data, to compute the gradient of W1
    xs = np.hstack([xs, np.ones((xs.shape[0], 1))])

    dW1 = (xs.T @ dh)/xs.shape[0]
    dW1 = dW1 + (_lambda_nn * model['W1']) # Add regularization
    return dict(W1=dW1, W2=dW2)

def get_gradient_nn(X_train, y_train, model):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward_nn(x, model)

        # Create one-hot coding of true label
        y_true = np.zeros(no_class)
        y_true[int(cls_idx)] = 1.

        # Compute the gradient of output layer
        #err = y_true - y_pred

        err =  y_pred - y_true

        # Accumulate the informations of the examples
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward_nn(np.array(xs), np.array(hs), np.array(errs), model)

def gradient_step_nn(X_train, y_train, model, lr):
    grad = get_gradient_nn(X_train, y_train, model)
    model = model.copy()
    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        model[layer] -= lr * grad[layer]

    return model

def find_accuracy_nn(X_train, y_train, model):
    y_pred = np.zeros_like(y_train)

    accuracy = 0

    for i, x in enumerate(X_train):
        # Predict the distribution of label
        #_, prob = forward(x, model)
        
        #my change
        _, prob = forward_nn(x, model)
        # Get label by picking the most probable one
        y = np.argmax(prob)
        y_pred[i] = y

    # Accuracy of predictions with the true labels and take the percentage
    # Because our dataset is balanced, measuring just the accuracy is OK
    accuracy = (y_pred == y_train).sum() / y_train.size
    #print ("Train accuracy", accuracy)
    macro_f1 = f1_score(y_train.tolist(), y_pred.tolist(), average='macro')
    return accuracy, macro_f1
    #return accuracy


def find_loss_nn(X_train, y_train, model):
    y_pred = np.zeros_like(y_train)
    #for i, x in enumerate(X_train):
    for x, cls_idx in zip(X_train, y_train):
        _, prob = forward_nn(x, model)
        # Create one-hot coding of true label
        y_true = np.zeros(no_class)
        y_true[int(cls_idx)] = 1.
    #loss = (-1 /  X_train.shape[0]) * np.sum(y_true * np.log(prob + EPSILON)) ###without regularization
    #loss = np.sum((-1 /  X_train.shape[0]) * np.sum(y_true * np.log(prob + EPSILON))) + ((_lambda_nn/2)* _l2_loss(model))  ###with regularization
    loss = (-1 /  X_train.shape[0]) * np.sum(y_true * np.log(prob + EPSILON)) + ((_lambda_nn/2)* _l2_loss(model)) 
    #print ("nll loss", loss)
    return loss
def get_test_label_nn(tf_idf_model, model):
    predictions = []
    for i, x in enumerate(tf_idf_model):
        #print (x)
        _, prob = forward_nn(x, model)
        #print(prob, prob.shape) ###(17,)
        pred = np.argmax(prob)
        #print (pred)
        predictions.append(pred-1)
    return predictions

def save_plots(accs_train_cv_lr,accs_val_cv_lr,f1_score_train_cv_lr,f1_score_val_cv_lr):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    #n = len(accs_train_cv_lr)
    #xs = np.arange(n)
    xs = learning_rate_nn

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, accs_train_cv_lr, '--', marker='o', linewidth=2, label='train')
    ax.plot(xs, accs_val_cv_lr, '--', marker='o', linewidth=2, label='validation')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy_nn_lc.png')

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, f1_score_train_cv_lr, '--', marker='o',linewidth=2, label='train')
    ax.plot(xs, f1_score_val_cv_lr, '--', marker='o', linewidth=2, label='validation')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Macro F1 Score")
    ax.legend(loc='lower right')
    plt.savefig('f1score_nn_lc.png')

def union_trainset(j, S1, S2 , S3 , S4 , S5 ):
    train_set_union = []
    #print "j",j
    if j == 1:
        test_set = pd.DataFrame(S1)
        #print "S1", S1, type(S1), type(test_set) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S2 , S3 , S4 , S5 ), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 1)
    if j == 2:
        test_set = pd.DataFrame(S2)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S3 , S4 , S5 ), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 1)
    if j == 3:
        test_set = pd.DataFrame(S3)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S4 , S5 ), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 1)
    if j == 4:
        test_set = pd.DataFrame(S4)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1, S2 , S3, S5 ), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 1)
    if j == 5:
        test_set = pd.DataFrame(S5)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2, S3 , S4), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 1)
    return train_set_union_df, test_set

def NN():

    # Read training data as dictionary
    #train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')
    df_nn = pd.read_csv("train.csv")
    
    #y_train = df_nn.label.values - 1 
    #print (y_train) # [ 3 15 10 ...  4 11  0]
    

    ######## 1-fold CV######

    # train = df_nn.sample(frac=0.7,random_state=2, replace = False) #random state is a seed value
    # validation = df_nn.drop(train.index)
    # y_train = train.label.values - 1
    # y_val = validation.label.values - 1

    ##### 5-fold CV########
    accs_train_cv_lr = []
    accs_val_cv_lr = []
    f1_score_train_cv_lr = []
    f1_score_val_cv_lr = []
    ##### for 5-fold cross validation ###########
    S1 = df_nn.values[0:246, :]
    S2 = df_nn.values[246:492, :]
    S3 = df_nn.values[492:738, :]
    S4 = df_nn.values[738:984, :]
    S5 = df_nn.values[984:1230, :]
    for lr in learning_rate_nn:
        accs_train_cv = []
        accs_val_cv = []
        f1_score_train_cv = []
        f1_score_val_cv = []
        for j in range (1,6):
            train , validation =  union_trainset(j, S1, S2 , S3 , S4 , S5 )
            train.columns = ['tweet_id', 'issue', 'text', 'author', 'label'] 
            validation.columns = ['tweet_id', 'issue', 'text', 'author', 'label'] 
            #print (train, type(train))
            #print (validation, type(validation))

            y_train = train.label.values -1
            #print("y_train", y_train, type(y_train))
            y_val = validation.label.values -1 

        
            ##########.....Train data pre-processing....... ######
            #training_data = pre_processing_tweets(df_nn)
            training_data = pre_processing_tweets(train)
            #print("training_data", type(training_data))
            ##########.....Build vocabulary on preprpcessed Train data....... ######
            vocab = build_vocabulary(training_data)

            ##########.....Build IDF on preprpcessed Train data....... ######
            word_idf_values = build_IDF(training_data, vocab)

            ##### Build TF-DF representation of train data ########
            tf_idf_model_train = build_representation(training_data, vocab, word_idf_values)



            ##########.....validation data pre-processing....... ######
            validation_data = pre_processing_tweets(validation)
            tf_idf_model_val = build_representation(validation_data, vocab, word_idf_values)



            ######## train MLP #############

            ######## Initialize NN weights #####
            #print("len vocab", len(vocab))
            model_nn = init_weights_nn(len(vocab))
            #print ("model_nn", model_nn)
            # print ("initial W1", model_nn["W1"])
            # print ("initial W2", model_nn["W2"])
            # print ("initial W3", model_nn["W3"])
            acc_nn_train = []
            loss_nn_train = []

            acc_nn_val = []
            loss_nn_val = []

            f1_score_train =[]
            f1_score_val = []

            for i in range(n_epochs):

                print('Iteration {}'.format(i))
                #print (X_train.shape) #(1230, 2500)
                model_nn = gradient_step_nn(tf_idf_model_train, y_train, model_nn, lr)
                ###find train accuracy
                acc, f1_score = find_accuracy_nn(tf_idf_model_train, y_train, model_nn )
                
                print ('Training Accuracy scratch: ', acc) ###using TF-IDF
                loss = find_loss_nn(tf_idf_model_train, y_train, model_nn )
                #print ('Training Loss scratch: ', loss) ###using TF-IDF    
                acc_nn_train.append(acc)
                loss_nn_train.append(loss)
                f1_score_train.append(f1_score)

                ###find validation accuracy
                acc_val, f1_score = find_accuracy_nn(tf_idf_model_val, y_val, model_nn )
                print ('validation Accuracy scratch: ', acc_val) ###using TF-IDF
                loss_val = find_loss_nn(tf_idf_model_val, y_val, model_nn )
                acc_nn_val.append(acc_val)
                loss_nn_val.append(loss_val)
                f1_score_val.append(f1_score)
                
            #save_plots(loss_nn_train, acc_nn_train, acc_nn_val, loss_nn_val)
            accs_train_cv.append(acc_nn_train[-1])
            f1_score_train_cv.append(f1_score_train[-1])
            accs_val_cv.append(acc_nn_val[-1]) 
            f1_score_val_cv.append(f1_score_val[-1])

        print  ("accs_train_cv", accs_train_cv)
        print("accs_val_cv", accs_val_cv)
        print  ("f1_score_train_cv", f1_score_train_cv)
        print  ("f1_score_val_cv", f1_score_val_cv)

        print  ("avg train acc", np.mean(accs_train_cv))
        print("avg validation acc", np.mean(accs_val_cv))
        print ("avg train f1-score", np.mean(f1_score_train_cv))
        print ("avg validation f1-score", np.mean(f1_score_val_cv))

        accs_train_cv_lr.append(np.mean(accs_train_cv))
        f1_score_train_cv_lr.append(np.mean(f1_score_train_cv))
        accs_val_cv_lr.append(np.mean(accs_val_cv))
        f1_score_val_cv_lr.append(np.mean(f1_score_val_cv))
        
    print ("accs_train_cv_lr", accs_train_cv_lr)
    print ("f1_score_train_cv_lr", f1_score_train_cv_lr)
    print ("accs_val_cv_lr", accs_val_cv_lr)
    print ("f1_score_val_cv_lr", f1_score_val_cv_lr)
    
    # accs_train_cv_lr =[0.16077235772357726, 0.26666666666666666, 0.39390243902439026, 0.9536585365853659, 0.9979674796747968]
    # f1_score_train_cv_lr =[0.043836118417034456, 0.08987695381454255, 0.1662815433543258, 0.8335217742973298, 0.9982861952979251]
    # accs_val_cv_lr =[0.15853658536585366, 0.15609756097560976, 0.20243902439024392, 0.39674796747967483, 0.41707317073170724]
    # f1_score_val_cv_lr =[0.04028808456310863, 0.04935513746669184, 0.0708958252038385, 0.24729115540563265, 0.29293708627620196]

    save_plots(accs_train_cv_lr, accs_val_cv_lr, f1_score_train_cv_lr, f1_score_val_cv_lr)

    ########,.......... Read test data
    #test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    df_test = pd.read_csv("test.csv") 
    #print (df.shape) #820,5
    #print (df.head())
    test_data = pre_processing_tweets(df_test)
    tf_idf_model_test = build_representation(test_data, vocab, word_idf_values)

    #### data normalization
    ####replace the elements of np array which is greater than maximum with maximum
    #arr[arr > 255] = x
    # tf_idf_model_test[tf_idf_model_test > maximum] = maximum
    # ####replace the elements of np array which is smaller than minimum with minimum
    # tf_idf_model_test[tf_idf_model_test < minimum] = minimum

    # tf_idf_model_test = data_normalization(tf_idf_model_test, minimum, maximum)

    # Predict test data by learned model
    test_pred = get_test_label_nn(tf_idf_model_test, model_nn)
    #print (test_pred, len(test_pred)) #820
    

    
    #Replace the following random predictor by your prediction function.
    

    #df_test = pd.DataFrame({'label': test_pred})
    df_test_new = pd.DataFrame({'tweet_id': df_test.tweet_id})
    df_test_new['issue'] = df_test.issue
    df_test_new['text'] = df_test.text
    df_test_new['author'] = df_test.author
    df_test_new['label'] = test_pred
    df_test_new.to_csv('test_nn_cv_wo_norm.csv')

if __name__ == '__main__':
    #LR()
    NN()