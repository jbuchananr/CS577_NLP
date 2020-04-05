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

n_epochs = 100
#learning_rate = .5
learning_rate = [.01,.05, .1, .5, 1]
_lambda = 0.1
no_class = 17
EPSILON = 1e-14
np.random.seed(1) ## seed

def init_weights(embedding_dim):
    #mu, sigma = 0, 0.00001
    mu, sigma = 0, 0.1
    model = dict ( W1 = np.random.normal(mu, sigma, [embedding_dim+1, no_class+1]))
    return model

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def stable_softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def sigmoid(z):
    return 1/(1+np.exp(-z))
def oneHotIt(Y):
    m = Y.shape[0]
    y_hot = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    y_hot = np.array(y_hot.todense()).T
    return y_hot
def forward(x, model):
    x = np.hstack([x, np.ones([x.shape[0], 1])]) #bias is in last column
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

    #print(y_true, y_true.shape)

    #compute gradient of output layer
    #err = y_true - y_pred

    err = y_pred - y_true
    return backward(X_train, err, model)

def gradient_step(X_train, y_train, model, lr): #define single gradient ascent step here we update parameters
    #print (X_train.shape) #(1230, 2500)
    grad = get_gradient (X_train, y_train, model) 

    #update every parameter of network using their gradients
    for param in grad:
        #print (param)
        #model[param] -= learning_rate * grad[param]
        model[param] -= lr * grad[param]
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
    #print (preds, preds.shape)
    count = 0
    for i in range (0, len(preds)):
        if (preds[i] == y_train[i]):
            count += 1
    #print ("count", count)
    accuracy = sum(preds == y_train)/(float(len(y_train)))
    macro_f1 = f1_score(y_train.tolist(), preds.tolist(), average='macro')
    return accuracy, macro_f1

def get_test_label(tf_idf_model, model):
    probs = forward(tf_idf_model, model)
    #print(probs, probs.shape) ###(1230, 18)
    preds = np.argmax(probs,axis=1)
    return preds

# def find_loss(tf_idf_model, y_train, model):
#     y_true = oneHotIt (y_train)
#     probs = forward(tf_idf_model, model)
#     loss = (-1 /  tf_idf_model.shape[0]) * np.sum(y_true * np.log(probs)) + (_lambda/2)*np.sum(model['W1']*model['W1']) 
#     #print ("loss", loss)
#     return loss


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
    loss = (-1 /  X_train.shape[0]) * np.sum(y_true * np.log(prob + EPSILON)) + ((_lambda/2)* _l2_loss(model))  ###with regularization
    #print ("nll loss", loss)
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

#def save_plots(losses, train_accs, val_accs, loss_nn_val):
def save_plots(accs_train_cv_lr,accs_val_cv_lr,f1_score_train_cv_lr,f1_score_val_cv_lr):
    """Plot

        Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    #n = len(accs_train_cv_lr)
    #xs = np.arange(n)
    xs = learning_rate

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, accs_train_cv_lr, '--', marker='o', linewidth=2, label='train')
    ax.plot(xs, accs_val_cv_lr, '--', marker='o', linewidth=2, label='validation')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy_lr.png')

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, f1_score_train_cv_lr, '--', marker='o',linewidth=2, label='train')
    ax.plot(xs, f1_score_val_cv_lr, '--', marker='o', linewidth=2, label='validation')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Macro F1 Score")
    ax.legend(loc='lower right')
    plt.savefig('f1score_lr.png')
def LR():

    # Read training data as dictionary
    df = pd.read_csv("train.csv") ###add lineterminator to avoid Error in Reading a csv file in pandas[CParserError: Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.]
    #print (df.shape) #1230
    #print (df.head())

    
    accs_train_cv_lr = []
    accs_val_cv_lr = []
    f1_score_train_cv_lr = []
    f1_score_val_cv_lr = []
    #### Five-fold Cross-validation
    #for cv in range (5):
   

    #train = df.sample(frac=0.8,random_state=1, replace = False) #random state is a seed value
    #validation = df.drop(train.index)
    #print (train.head) #984 rows x 5 columns
    #y_train = train.label.values 
    #print("y_train", type(y_train))
    #y_val = validation.label.values 
    #print("y_val", y_val)

    ##### for 5-fold cross validation ###########
    S1 = df.values[0:246, :]
    S2 = df.values[246:492, :]
    S3 = df.values[492:738, :]
    S4 = df.values[738:984, :]
    S5 = df.values[984:1230, :]
    for lr in learning_rate:
        accs_train_cv = []
        accs_val_cv = []
        f1_score_train_cv = []
        f1_score_val_cv = []

        for j in range (1,6):
            accs = []
            losses = []
            accs_val = []
            losses_val = []
            f1_score_train =[]
            f1_score_val = []

            train , validation =  union_trainset(j, S1, S2 , S3 , S4 , S5 )
            train.columns = ['tweet_id', 'issue', 'text', 'author', 'label'] 
            validation.columns = ['tweet_id', 'issue', 'text', 'author', 'label'] 
            #print (train, type(train))
            #print (validation, type(validation))

            y_train = train.label.values 
            #print("y_train", y_train, type(y_train))
            y_val = validation.label.values 


            ##########.....Train data pre-processing....... ######
            training_data = pre_processing_tweets(train)
            #print("training_data", type(training_data))
            ##########.....Build vocabulary on preprpcessed Train data....... ######
            vocab = build_vocabulary(training_data)

            ##########.....Build IDF on preprpcessed Train data....... ######
            word_idf_values = build_IDF(training_data, vocab)

            ##### Build TF-DF representation of train data ########
            tf_idf_model_train = build_representation(training_data, vocab, word_idf_values)
            
            #### train data normalization
            # minimum = np.min(tf_idf_model_train)
            # maximum = np.max(tf_idf_model_train)
            # tf_idf_model_train = data_normalization(tf_idf_model_train, minimum, maximum)

            ######## sklearn ##########

            #clf = LogisticRegression( solver='lbfgs', random_state=1, max_iter = 100,  multi_class='auto', fit_intercept = True).fit(tf_idf_model_train, y_train.astype('int') )
            #print("train accuracy skitlearn",clf.score(tf_idf_model_train, y_train.astype('int'))) #55%, w/o normalization 98%


            ##########.....validation data pre-processing....... ######
            validation_data = pre_processing_tweets(validation)
            tf_idf_model_val = build_representation(validation_data, vocab, word_idf_values)

            #### validation data normalization
            ####replace the elements of np array which is greater than maximum with maximum
            #arr[arr > 255] = x
            # tf_idf_model_val[tf_idf_model_val > maximum] = maximum
            # ####replace the elements of np array which is smaller than minimum with minimum
            # tf_idf_model_val[tf_idf_model_val < minimum] = minimum

            # tf_idf_model_val = data_normalization(tf_idf_model_val, minimum, maximum)
            
            #print("validation accuracy skitlearn",clf.score(tf_idf_model_val, y_val.astype('int'))) #44%, w/o normalization 96%

            #print ("len vocab",len(vocab)) #2793
            model = init_weights(len(vocab))

            for i in range(n_epochs):

                print('Iteration {}'.format(i))
                #print (X_train.shape) #(1230, 2500)
                #model = gradient_step(tf_idf_model_train, y_train, model)
                model = gradient_step(tf_idf_model_train, y_train, model, lr)

                ###find train accuracy
                #print ('Training Accuracy prev: ', find_accuracy(sentence_vectors, y_train, model )) ###using BOW
                acc, f1_score = find_accuracy(tf_idf_model_train, y_train, model )
                print ('Training Accuracy scratch: ', acc) ###using TF-IDF
                accs.append(acc)
                f1_score_train.append(f1_score)
                ###find train Loss
                loss = find_loss(tf_idf_model_train, y_train, model )
                print ('Training Loss scratch: ', loss) ###using TF-IDF 
                losses.append(loss) 


                ###find validation accuracy
                acc_val, f1_score = find_accuracy(tf_idf_model_val, y_val, model )
                print ('validation Accuracy scratch: ', acc_val) ###using TF-IDF
                accs_val.append(acc_val)
                f1_score_val.append(f1_score)
                ###find validation Loss
                loss_val = find_loss(tf_idf_model_val, y_val, model )
                print ('validation Loss scratch: ', loss_val) ###using TF-IDF 
                # losses_val.append(loss_val) 
            accs_train_cv.append(accs[-1])
            f1_score_train_cv.append(f1_score_train[-1])
            accs_val_cv.append(accs_val[-1])  
            f1_score_val_cv.append(f1_score_val[-1])
        print  ("accs_train_cv", accs_train_cv)
        print  ("f1_score_train_cv", f1_score_train_cv)
        print  ("accs_val_cv", accs_val_cv)
        print  ("f1_score_val_cv", f1_score_val_cv)

        print  ("avg train acc", np.mean(accs_train_cv))
        print ("avg train f1-score", np.mean(f1_score_train_cv))
        print("avg validation acc", np.mean(accs_val_cv))
        print ("avg validation f1-score", np.mean(f1_score_val_cv))

        accs_train_cv_lr.append(np.mean(accs_train_cv))
        f1_score_train_cv_lr.append(np.mean(f1_score_train_cv))
        accs_val_cv_lr.append(np.mean(accs_val_cv))
        f1_score_val_cv_lr.append(np.mean(f1_score_val_cv))

    print ("accs_train_cv_lr", accs_train_cv_lr)
    print ("f1_score_train_cv_lr", f1_score_train_cv_lr)
    print ("accs_val_cv_lr", accs_val_cv_lr)
    print ("f1_score_val_cv_lr", f1_score_val_cv_lr)
    save_plots(accs_train_cv_lr,accs_val_cv_lr,f1_score_train_cv_lr,f1_score_val_cv_lr)

    ###########.........Test dat .........########
    #test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')
    df_test = pd.read_csv("test.csv") 
    #print (df_test.shape) #820,5
    #print (df.head())
    test_data = pre_processing_tweets(df_test)
    tf_idf_model_test = build_representation(test_data, vocab, word_idf_values)

    #### test data normalization
    # ####replace the elements of np array which is greater than maximum with maximum
    # tf_idf_model_test[tf_idf_model_test > maximum] = maximum
    # ####replace the elements of np array which is smaller than minimum with minimum
    # tf_idf_model_test[tf_idf_model_test < minimum] = minimum

    # tf_idf_model_test = data_normalization(tf_idf_model_test, minimum, maximum)

    # Predict test data by learned model
    test_pred = get_test_label(tf_idf_model_test, model)
    #print (test_pred, len(test_pred)) #820
    
    #df_test = pd.DataFrame({'label': test_pred})
    df_test_new = pd.DataFrame({'tweet_id': df_test.tweet_id})
    df_test_new['issue'] = df_test.issue
    df_test_new['text'] = df_test.text
    df_test_new['author'] = df_test.author
    df_test_new['label'] = test_pred
    #df_test_new.to_csv('test_lr.csv')
   

def NN():

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Neural Network classifier here
    '''

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')

    # Predict test data by learned model
    # Replace the following random predictor by your prediction function

    for tweet_id in test_tweet_id2text:
        # Get the text
        text=test_tweet_id2text[tweet_id]

        # Predict the label
        label=randrange(1, 18)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id]=label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_nn.csv')

if __name__ == '__main__':
    LR()
    #NN()