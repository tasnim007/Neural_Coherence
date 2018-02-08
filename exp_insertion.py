from __future__ import division
# from keras.models import load_model
# from keras.layers import Flatten, Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
# from keras.models import Model
# from keras import objectives
# from keras.preprocessing import sequence
# from keras.callbacks import ModelCheckpoint

import numpy as np
from keras.utils import np_utils
from keras import backend as K

from utilities import data_helper
from utilities import my_callbacks

import sys
import tensorflow as tf
import optparse

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

def forward_propagation(X_positive, X_negative, vocab, E, mode, print_=False):
    ## First Layer of NN: Transform each grammatical role in the grid into distributed representation - a real valued vector

    # Shared embedding matrix
    # W_embedding = tf.get_variable("W_embedding", [len(vocab), 100], initializer = tf.contrib.layers.xavier_initializer()) #embedding matrix
    # E = np.float32(E) # DataType of E is float64, which is not in list of allowed values in conv1D. Allowed DataType: float16, float32
    E = tf.convert_to_tensor(E, tf.float32)
    W_embedding = tf.get_variable("W_embedding", initializer=E)  # embedding matrix

    # Look up layer

    # for positive document
    embedding_positive = tf.nn.embedding_lookup(W_embedding, X_positive)

    # for negative document
    embedding_negative = tf.nn.embedding_lookup(W_embedding, X_negative)

    ## Second Layer of NN: Convolution Layer

    # shared filter and bias
    # w_size = 6       #filter_size
    # emb_size = 100   #embedding_size
    # nb_filter = 150  #num_filters

    filter_sizes = []
    if opts.filter_list != "":  # stupid arge parsing, do it latter
        for i in opts.filter_list.split(","):
            filter_sizes.append(int(i))

    pooled_outputs_positive = []
    pooled_outputs_negative = []
    #filter_sizes = [8, 9, 10]
    for i, w_size in enumerate(filter_sizes):

        filter_shape = [w_size, opts.emb_size, opts.nb_filter]

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)  # l2 regularizer for filter

        #W_conv_layer_1 = tf.get_variable("W_conv_layer_1", shape = filter_shape, initializer = tf.contrib.layers.xavier_initializer(seed = 0)) #filter for covolution layer 1

        #W_conv_layer_1 = tf.get_variable("W_conv_layer_1", shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer(seed=2018), regularizer=regularizer)  # filter for covolution layer 1
        #b_conv_layer_1 = tf.get_variable("b_conv_layer_1", shape=[opts.nb_filter], initializer=tf.constant_initializer(0.0))  # bias for convolution layer 1

        initializer = tf.contrib.layers.xavier_initializer(seed=opts.seed)
        W_conv_layer_1 = tf.Variable(initializer(filter_shape), name="W_conv_layer_1")
        b_conv_layer_1 = tf.Variable(tf.constant(0.0, shape=[opts.nb_filter]), name="b_conv_layer_1")

        # 1D Convolution for positive document
        conv_layer_1_positive = tf.nn.conv1d(embedding_positive, W_conv_layer_1, stride=1,
                                             padding="VALID")  # embedding and W_conv_layer_1 both are 3D matrix
        conv_layer_1_with_bias_positive = tf.nn.bias_add(conv_layer_1_positive, b_conv_layer_1)

        conv_layer_1_with_bn_positive = tf.layers.batch_normalization(conv_layer_1_with_bias_positive,
                                                                      axis=2,
                                                                      center=True,
                                                                      scale=False,
                                                                      training=(mode == tf.estimator.ModeKeys.TRAIN)
                                                                      )

        h_conv_layer_1_positive = tf.nn.relu(conv_layer_1_with_bn_positive,
                                             name="relu_conv_layer_1_positive")  # Apply nonlinearity

        # 1D Convolution for negative document
        conv_layer_1_negative = tf.nn.conv1d(embedding_negative, W_conv_layer_1, stride=1,
                                             padding="VALID")  # embedding and W_conv_layer_1 both are 3D matrix
        conv_layer_1_with_bias_negative = tf.nn.bias_add(conv_layer_1_negative, b_conv_layer_1)

        conv_layer_1_with_bn_negative = tf.layers.batch_normalization(conv_layer_1_with_bias_negative,
                                                                      axis=2,
                                                                      center=True,
                                                                      scale=False,
                                                                      training=(mode == tf.estimator.ModeKeys.TRAIN)
                                                                      )

        h_conv_layer_1_negative = tf.nn.relu(conv_layer_1_with_bn_negative,
                                             name="relu_conv_layer_1_negative")  # Apply nonlinearity

        ## Third Layer of NN: Pooling Layer

        # maxpooling

        # 1D Pooling for positive document
        #window_shape=[opts.pool_length],
        #strides=[opts.pool_length],
        m_layer_1_positive_inside = tf.nn.pool(h_conv_layer_1_positive, window_shape=[w_size],
                                        strides=[w_size],
                                        pooling_type='MAX', padding="VALID")

        # 1D Pooling for negative document
        m_layer_1_negative_inside = tf.nn.pool(h_conv_layer_1_negative, window_shape=[w_size],
                                        strides=[w_size],
                                        pooling_type='MAX', padding="VALID")

 
        pooled_outputs_positive.append(m_layer_1_positive_inside)
        pooled_outputs_negative.append(m_layer_1_negative_inside)

    m_layer_1_positive = tf.concat(pooled_outputs_positive, 1)
    m_layer_1_negative = tf.concat(pooled_outputs_negative, 1)

    # for positive document
    flatten_positive = tf.contrib.layers.flatten(m_layer_1_positive)
    # flatten_positive = tf.contrib.layers.flatten(drop_out_early_positive)

    # for negative document
    flatten_negative = tf.contrib.layers.flatten(m_layer_1_negative)
    # flatten_negative = tf.contrib.layers.flatten(drop_out_early_negative)

    # Dropout

    # for positive document
    drop_out_positive = tf.nn.dropout(flatten_positive, keep_prob=opts.dropout_ratio, seed=opts.seed)

    # for negative document
    drop_out_negative = tf.nn.dropout(flatten_negative, keep_prob=opts.dropout_ratio, seed=opts.seed)

    # Coherence Scoring
    dim_coherence = drop_out_positive.shape[1]
    v_fc_layer = tf.get_variable("v_fc_layer", shape=[dim_coherence, 1],
                                 initializer=tf.contrib.layers.xavier_initializer(
                                     seed=opts.seed))  # Weight matrix for final layer
    b_fc_layer = tf.get_variable("b_fc_layer", shape=[1],
                                 initializer=tf.constant_initializer(0.0))  # bias for final layer

    # for positive document
    # out_positive = tf.contrib.layers.fully_connected(drop_out_positive, num_outputs = 1, activation_fn=None)
    # out_positive = tf.sigmoid(out_positive)
    out_positive = tf.add(tf.matmul(drop_out_positive, v_fc_layer), b_fc_layer)

    # for negative document
    # out_negative = tf.contrib.layers.fully_connected(drop_out_negative, num_outputs = 1, activation_fn=None)
    # out_negative = tf.sigmoid(out_negative)
    out_negative = tf.add(tf.matmul(drop_out_negative, v_fc_layer), b_fc_layer)

    parameters = {"W_embedding": W_embedding,
                  "W_conv_layer_1": W_conv_layer_1,
                  "b_conv_layer_1": b_conv_layer_1,
                  "v_fc_layer": v_fc_layer,
                  "b_fc_layer": b_fc_layer}

    return out_positive, out_negative, parameters

#print("Starting neural grid... 25.01.18 ...20:45")
parser = optparse.OptionParser("%prog [options]")

# file related options
# [chijkqruvxyz] [ABDEGHIJKMNOQRSTUVWXYZ]
parser.add_option("-g", "--log-file", dest="log_file", help="log file [default: %default]")
parser.add_option("-d", "--data-dir", dest="data_dir",
                  help="directory containing list of train, test and dev file [default: %default]")
parser.add_option("-m", "--model-dir", dest="model_dir",
                  help="directory to save the best models [default: %default]")

parser.add_option("-t", "--max-length", dest="maxlen", type="int",
                  help="maximul length (for fixed size input) [default: %default]")  # input size
parser.add_option("-f", "--nb_filter", dest="nb_filter", type="int",
                  help="nb of filter to be applied in convolution over words [default: %default]")
# parser.add_option("-r", "--filter_length",    dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]")
parser.add_option("-w", "--w_size", dest="w_size", type="int",
                  help="window size length of neighborhood in words [default: %default]")
parser.add_option("-p", "--pool_length", dest="pool_length", type="int",
                  help="length for max pooling [default: %default]")
parser.add_option("-e", "--emb-size", dest="emb_size", type="int",
                  help="dimension of embedding [default: %default]")
parser.add_option("-s", "--hidden-size", dest="hidden_size", type="int",
                  help="hidden layer size [default: %default]")
parser.add_option("-o", "--dropout_ratio", dest="dropout_ratio", type="float",
                  help="ratio of cells to drop out [default: %default]")

parser.add_option("-a", "--learning-algorithm", dest="learn_alg",
                  help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
parser.add_option("-b", "--minibatch-size", dest="minibatch_size", type="int",
                  help="minibatch size [default: %default]")
parser.add_option("-l", "--loss", dest="loss",
                  help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
parser.add_option("-n", "--epochs", dest="epochs", type="int", help="nb of epochs [default: %default]")
parser.add_option("-P", "--permutation", dest="p_num", type="int", help="nb of permutation[default: %default]")
parser.add_option("-F", "--feats", dest="f_list",
                  help="semantic features using in the model, separate by . [default: %default]")
parser.add_option("-S", "--seed", dest="seed", type="int", help="seed for random number. [default: %default]")
parser.add_option("-C", "--margin", dest="margin", type="int",
                  help="margin of the ranking objective. [default: %default]")
parser.add_option("-M", "--eval_minibatches", dest="eval_minibatches", type="int",
                  help="How often we want to evaluate in an epoch. [default: %default]")
parser.add_option("-L", "--filter_list", dest="filter_list", help="List of filter sizes in conv layer. [default: %default]")

#parameters for insertion task
parser.add_option("-D", "--doc_list", dest="docs_list", help="list of test documents [default: %default]")
parser.add_option("-k", "--checkpoint", dest="checkpoint", help="checkpoint - saved model [default: %default]")

parser.set_defaults(

    data_dir="./data/"
    , log_file="log"
    , model_dir="./saved_models/gridCNN_Multifilter/"
    , learn_alg="rmsprop"  # sgd, adagrad, rmsprop, adadelta, adam (default)
    , loss="ranking_loss"  # hinge, squared_hinge, binary_crossentropy (default)
    , minibatch_size=10
    , dropout_ratio=1
    , maxlen=25000
    , epochs=10
    , emb_size=100
    , hidden_size=250
    , nb_filter=150
    , w_size=10
    , pool_length=10
    , p_num=20
    , f_list=""  # "0.1.3"
    , seed=2018
    , margin=6
    , eval_minibatches=100
    , filter_list="9,10,11"
    #parameters for insertion task
    , doc_list="./data/wsj.4test"
    , checkpoint= "./saved_models/gridCNN_Multifilter/gridCNN_epoch_0_minibatch_399"
)

opts, args = parser.parse_args(sys.argv)
print("------------------------------------------------------------")    
print(" -doc list:\t", opts.doc_list)
print(" -checkpoint:\t", opts.checkpoint)
print(" -maxlen:\t", opts.maxlen)
print(" -emb_size:\t", opts.emb_size)
print(" -hidden_size:\t", opts.hidden_size)
print(" -nb_filter:\t", opts.nb_filter)
print(" -filter_list :\t", opts.filter_list)
print("------------------------------------------------------------")    


print('Loading vocab of the whole dataset...')
vocab = data_helper.load_all(filelist="data/wsj.all")
print(vocab)

# create embeding
E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab), opts.emb_size))
E[len(vocab)-1] = 0
    
X_pos = tf.placeholder(tf.int32, shape=[None, opts.maxlen])  # Placeholder for positive document
X_neg = tf.placeholder(tf.int32, shape=[None, opts.maxlen])  # Placeholder for negative document
mode = tf.placeholder(tf.bool, name='mode')  # Placeholder needed for batch normalization

#build_graph 
score_positive, score_negative, parameters = forward_propagation(X_pos, X_neg, vocab, E, mode, print_=True)
# rescore checkpoint
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver.restore(sess, opts.checkpoint)
    

#find the maximum coherence score when inserting the sentence at position k
def insert(filename="", k = 0, w_size=3, maxlen=14000, vocab_list=None, fn=None):
    lines = [line.rstrip('\n') for line in open(filename+ ".EGrid")]
    doc_size = data_helper.find_len(sent=lines[1])
    X_1 =  data_helper.load_POS_EGrid(filename=filename, w_size=opts.w_size, maxlen=opts.maxlen , vocab_list=vocab)

    #print(X_1)
    #the lowest coherence score of a document
    bestScore = -999999.999999
    bestPos = []

    perm = []
    perm.append(k)
    for i in range(0, doc_size):
        if i!=k:
            perm.append(i)

    for pos in range(0,doc_size):
        #compute coherence score for permuated         
        X_0 =  data_helper.load_NEG_EGrid(filename=filename, w_size=opts.w_size , maxlen=opts.maxlen , vocab_list=vocab, perm=perm)
        #print(perm)
        #print(X_0)
        score_pos, score_neg = sess.run([score_positive, score_negative], feed_dict={X_pos: X_1, X_neg: X_0, mode: True})
        score_pos = score_pos[0][0]
        score_neg = score_neg[0][0]
 
        print(" - At position " + str(pos) + " |--> pos vs. neg score: " + str("%0.4f" % score_pos) + " vs. " + str("%0.4f" % score_neg) )
        #if score_neg >= score_pos: # bad insertion, we want score_1 is always greater than score_0
        if(score_neg > bestScore):

            bestScore = score_neg
            bestPos = []
            bestPos.append(pos)
        elif score_neg == bestScore:
            bestPos.append(pos)                
        
        if pos < doc_size-1:
            perm[pos] = perm[pos+1]
            perm[pos+1] = k
        #print(bestScore)

    return bestPos


totalPerf = 0
totalIns = 0 
docAvgPerf = 0.0

#main function here
list_of_files = [line.rstrip('\n') for line in open( opts.doc_list)]
totalPerf = 0
for file in list_of_files:
    # process each test document
    doc_size = data_helper.find_doc_size(file+".EGrid");
    print("------------------------------------------------------------")    
    print(str(file))    
    
    perfects = 0;
    for k in range(0, doc_size):
        print ("Insert sent " + str(k) + "...")
        bestPos = insert(file, k, w_size=opts.w_size, maxlen=opts.maxlen,vocab_list=vocab)

        print ("==> Having best coherrent positions: " + str(bestPos))
        if k in bestPos:
            perfects = perfects + 1

    totalPerf = totalPerf + perfects
    totalIns = totalIns + doc_size
    docAvgPerf = docAvgPerf + perfects / doc_size;
    print ("Document perfect: " + str(perfects) + " of " + str(doc_size))

print ("\nSummary...")  
print (" -Perfect: " + str(totalPerf)) 
print (" -Perfect by line: " + str(totalPerf/totalIns))    
print (" -Perfect by doc: " + str(docAvgPerf/len(list_of_files)))    

sess.close() # closing sess

