import tensorflow as tf
import numpy as np
#from utilities import my_callbacks
from utilities import data_helper
from utilities import new_data_helper
import optparse
import sys
import math

#np.set_printoptions(threshold=np.nan)


def forward_propagation(X_positive, X_negative, vocab_info, E, mode, print_ = False):
    """
    Implements forward propagation of Neural coherence model

    Arguments:
    X_positive -- A Placeholder for positive document
    X_negative -- A Placeholder for negative document
    vocab_info -- Dictionary of Vocabulary info
    E -- initialized values for embedding matrix
    mode -- whether we are in training: mode=True or testing: mode=False [Used in Batch Normalization].
    print_ -- Whether size of the variables to be printed

    Returns:
    out_positive -- Coherence Score for positive document
    out_negative -- Coherence Score for negative document
    parameters -- a dictionary of tensors containing trainable parameters

    """

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

    filter_shape = [opts.w_size, opts.emb_size, opts.nb_filter]

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)  #l2 regularizer for filter

    # W_conv_layer_1 = tf.get_variable("W_conv_layer_1", shape = filter_shape, initializer = tf.contrib.layers.xavier_initializer(seed = 0)) #filter for covolution layer 1
    W_conv_layer_1 = tf.get_variable("W_conv_layer_1", shape=filter_shape,
                                     initializer=tf.contrib.layers.xavier_initializer(
                                         seed=2018), regularizer=regularizer)  # filter for covolution layer 1
    b_conv_layer_1 = tf.get_variable("b_conv_layer_1", shape=[opts.nb_filter],
                                     initializer=tf.constant_initializer(0.0))  # bias for convolution layer 1

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

    # 1D Pooling for positive document
    m_layer_1_positive = tf.nn.pool(h_conv_layer_1_positive, window_shape=[opts.pool_length], strides=[opts.pool_length],
                                    pooling_type='MAX', padding="VALID")

    # 1D Pooling for negative document
    m_layer_1_negative = tf.nn.pool(h_conv_layer_1_negative, window_shape=[opts.pool_length], strides=[opts.pool_length],
                                    pooling_type='MAX', padding="VALID")

    ## Fourth Layer of NN: Fully Connected Layer

    # Dropout Early [As Dat Used]

    # for positive document
    # drop_out_early_positive = tf.nn.dropout(m_layer_1_positive, keep_prob=0.5)

    # for negative document
    # drop_out_early_negative = tf.nn.dropout(m_layer_1_negative, keep_prob=0.5)

    # Flatten

    # for positive document
    flatten_positive = tf.contrib.layers.flatten(m_layer_1_positive)
    # flatten_positive = tf.contrib.layers.flatten(drop_out_early_positive)

    # for negative document
    flatten_negative = tf.contrib.layers.flatten(m_layer_1_negative)
    # flatten_negative = tf.contrib.layers.flatten(drop_out_early_negative)

    # Dropout

    # for positive document
    drop_out_positive = tf.nn.dropout(flatten_positive, keep_prob=opts.dropout_ratio, seed=2018)

    # for negative document
    drop_out_negative = tf.nn.dropout(flatten_negative, keep_prob=opts.dropout_ratio, seed=2018)

    # Coherence Scoring
    dim_coherence = drop_out_positive.shape[1]
    v_fc_layer = tf.get_variable("v_fc_layer", shape=[dim_coherence, 1],
                                 initializer=tf.contrib.layers.xavier_initializer(
                                     seed=2018))  # Weight matrix for final layer
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

    if (print_):
        print("Layer (type)          Output Shape")
        print("_________________________________________")
        print("\nInputLayer:")
        print("X_positive           ", X_positive.shape)
        print("X_negative           ", X_negative.shape)
        print("\nEmbedding Layer:")
        print("Embedding Matrix     ", W_embedding.shape)
        print("Embedding Positive   ", embedding_positive.shape)
        print("Embedding Negative   ", embedding_negative.shape)
        print("\nConvolution 1D Layer:")
        print("Filter Shape         ", W_conv_layer_1.shape)
        print("Conv Positive        ", h_conv_layer_1_positive.shape)
        print("Conv Negative        ", h_conv_layer_1_negative.shape)
        print("\nMax Pooling 1D Layer:")
        print("MaxPool Positive     ", m_layer_1_positive.shape)
        print("MaxPool Negative     ", m_layer_1_negative.shape)
        print("\nFlatten Layer: ")
        print("Flatten Positive     ", flatten_positive.shape)
        print("Flatten Negative     ", flatten_negative.shape)
        print("\nDropout Layer: ")
        print("Dropout Positive     ", drop_out_positive.shape)
        print("Dropout Negative     ", drop_out_negative.shape)
        print("\nFully Connected Layer:")
        print("FC Positive          ", out_positive.shape)
        print("FC Negative          ", out_negative.shape)

    return out_positive, out_negative, parameters


def ranking_loss(pos, neg):
    """
    Implements the ranking objective.

    Arguments:
    pos -- score for positive document batch
    neg -- score for negative document batch

    Returns:
    Average ranking loss for the batch

    """

    loss = tf.maximum(opts.margin + neg - pos, 0.0)
    # print(loss)
    return tf.reduce_mean(loss)


def mini_batches(X, Y, mini_batch_size=32, shuffle=False):
    """
    Creates a list of minibatches from (X, Y)

    Arguments:
    X -- Positive Documents
    Y -- Negative Documents
    mini_batch_size -- Size of each mini batch
    shuffle -- whether to shuffle the data before creating minibatches

    Returns:
    list of mini batches from the positive and negative documents.

    """
    m = X.shape[0]
    mini_batches = []

    if(shuffle):
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]
    else:
        shuffled_X = X
        shuffled_Y = Y


    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



if __name__ == '__main__':
    # parse user input
    print("Starting...9:12")
    parser = optparse.OptionParser("%prog [options]")

    #file related options
    #[chijkqruvxyz] [ABDEGHIJKLMNOQRSUVWXYZ]
    parser.add_option("-g", "--log-file",   dest="log_file", help="log file [default: %default]")
    parser.add_option("-d", "--data-dir",   dest="data_dir", help="directory containing list of train, test and dev file [default: %default]")
    parser.add_option("-m", "--model-dir",  dest="model_dir", help="directory to save the best models [default: %default]")

    parser.add_option("-t", "--max-length",        dest="maxlen", type="int", help="maximul length (for fixed size input) [default: %default]") # input size
    parser.add_option("-f", "--nb_filter",         dest="nb_filter",     type="int",   help="nb of filter to be applied in convolution over words [default: %default]")
    #parser.add_option("-r", "--filter_length",    dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]")
    parser.add_option("-w", "--w_size",            dest="w_size", type="int",   help="window size length of neighborhood in words [default: %default]")
    parser.add_option("-p", "--pool_length",       dest="pool_length",   type="int",   help="length for max pooling [default: %default]")
    parser.add_option("-e", "--emb-size",          dest="emb_size",      type="int",   help="dimension of embedding [default: %default]")
    parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int",   help="hidden layer size [default: %default]")
    parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

    parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
    parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
    parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
    parser.add_option("-n", "--epochs",       dest="epochs", type="int", help="nb of epochs [default: %default]")
    parser.add_option("-P", "--permutation",  dest="p_num", type="int", help="nb of permutation[default: %default]")
    parser.add_option("-F", "--feats",        dest="f_list", help="semantic features using in the model, separate by . [default: %default]")
    parser.add_option("-S", "--seed",         dest="seed", type="int", help="seed for random number. [default: %default]")
    parser.add_option("-C", "--margin",       dest="margin", type="int", help="margin of the ranking objective. [default: %default]")
    parser.add_option("-M", "--eval_minibatches", dest="eval_minibatches", type="int",
                      help="How often we want to evaluate in an epoch. [default: %default]")
    parser.add_option("-T", "--pretrained", dest="pretrained", type="int", help="Whether we are going to use pretrained word embedding. [default: %default]")
    parser.set_defaults(

        data_dir="./data/"
        , log_file="log"
        , model_dir="./saved_models/"

        , learn_alg="rmsprop"  # sgd, adagrad, rmsprop, adadelta, adam (default)
        , loss="ranking_loss"  # hinge, squared_hinge, binary_crossentropy (default)
        , minibatch_size=10
        , dropout_ratio=1

        , maxlen=25000
        , epochs=5
        , emb_size=300
        , hidden_size=250
        , nb_filter=150
        , w_size=10
        , pool_length=6
        , p_num=20
        , f_list=""  # "0.1.3"

        , seed=2018
        , margin=6
        , eval_minibatches=100
        , pretrained=1
    )

    opts, args = parser.parse_args(sys.argv)

    print("\n\n**Hyperparameters**")
    print("minibatch_size: ", opts.minibatch_size, "  dropout_ratio: ", opts.dropout_ratio,
          "  maxlen: ", opts.maxlen, "  epochs: ", opts.epochs, "  emb_size: ", opts.emb_size, "  hidden_size: ",
          opts.hidden_size, "  nb_filter: ", opts.nb_filter, "  w_size: ", opts.w_size,
          "  pool_length: ", opts.pool_length, "  p_num: ", opts.p_num, "  seed: ", opts.seed,
          "  margin: ", opts.margin, "  eval_minibatches: ", opts.eval_minibatches)

    print('Loading vocab of the whole dataset...')
    entity_type = ['S', 'O', 'X', '-', '0']
    vocabs = new_data_helper.init_vocab(filelist="./data/wsj.train_dev", occur=90)
    print("Len of vocabs: ",len(vocabs))


    print("loading entity-grid for pos and neg documents...")
    X_train_1, X_train_0, vocab_info_train = new_data_helper.load_and_numberize_egrids_word_entity(filelist="./data/wsj.train",
        maxlen=opts.maxlen, w_size=opts.w_size, vocabs=vocabs)

    X_dev_1, X_dev_0, vocab_info_dev = new_data_helper.load_and_numberize_egrids_word_entity(filelist="./data/wsj.dev",
        maxlen=opts.maxlen, w_size=opts.w_size, vocabs=vocabs)

    X_test_1, X_test_0, vocab_info_test = new_data_helper.load_and_numberize_egrids_word_entity(filelist="./data/wsj.test",
        maxlen=opts.maxlen, w_size=opts.w_size, vocabs=vocabs)


    ###########Pretrained word embedding
    if opts.pretrained == 1:
        vocab_0 = []
        for ent in vocabs:
            vocab_0.append(ent)
            vocab_0.append(ent)
            vocab_0.append(ent)

        vocab_0.append("-")
        vocab_0.append("0")

        E = data_helper.load_pretrained_glove(vocabs=vocab_0, filename="./glove/glove.6B.300d.txt")
        print("Shape of pretrained glove: ", E.shape)

    else:
        vocab_x = vocab_info_train["vocab_x"]
        np.random.seed(2018)
        E = 0.01 * np.random.uniform(-1.0, 1.0, (len(vocab_x), opts.emb_size))
        E[len(vocabs) - 1] = 0

    ###########################

    '''
    X_train_1 = np.concatenate((X_train_1, X_dev_1), axis=0)
    X_train_0 = np.concatenate((X_train_0, X_dev_0), axis=0)
    '''
    num_train = len(X_train_1)
    num_dev   = len(X_dev_1)
    num_test  = len(X_test_1)
    #assign Y value
    y_train_1 = [1] * num_train
    y_dev_1 = [1] * num_dev
    y_test_1 = [1] * num_test

    print('.....................................')
    print("Num of train pairs: " + str(num_train))
    print("Num of dev pairs: " + str(num_dev))
    print("Num of test pairs: " + str(num_test))
    #print("Num of permutation in train: " + str(opts.p_num))
    #print("The maximum in length for CNN: " + str(opts.maxlen))
    print('.....................................')


    #randomly shuffle the training data
    np.random.seed(113)
    np.random.shuffle(X_train_1)
    np.random.seed(113)
    np.random.shuffle(X_train_0)


    ## Create Placeholders
    X_positive = tf.placeholder(tf.int32, shape = [None, opts.maxlen]) #Placeholder for positive document
    X_negative = tf.placeholder(tf.int32, shape = [None, opts.maxlen]) #Placeholder for negative document
    mode = tf.placeholder(tf.bool, name='mode')  #Placeholder needed for batch normalization




    # Forward propagation
    score_positive, score_negative, parameters = forward_propagation(X_positive, X_negative, vocab_info_train, E, mode, print_=True)



    # Cost function:
    cost = ranking_loss(score_positive, score_negative)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.0, momentum=0.9, epsilon=1e-8).minimize(cost)


    ## Using keras RMSProp

    W_embedding = parameters["W_embedding"]
    W_conv_layer_1 = parameters["W_conv_layer_1"]
    b_conv_layer_1 = parameters["b_conv_layer_1"]
    v_fc_layer = parameters["v_fc_layer"]
    b_fc_layer = parameters["b_fc_layer"]
    optimizer = tf.keras.optimizers.RMSprop().get_updates(cost, [W_embedding, W_conv_layer_1, b_conv_layer_1, v_fc_layer, b_fc_layer])



    init = tf.global_variables_initializer()

    m = num_train

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(opts.epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / opts.minibatch_size) # number of minibatches of size minibatch_size in the train set
            #minibatches = random_mini_batches(X_train_1, X_train_0, opts.minibatch_size)
            minibatches = mini_batches(X_train_1, X_train_0, opts.minibatch_size)

            for (i, minibatch) in enumerate(minibatches):

                (minibatch_X_positive, minibatch_X_negative) = minibatch

                _, temp_cost, pos, neg = sess.run([optimizer, cost, score_positive, score_negative],
                            feed_dict={X_positive:minibatch_X_positive,
                                    X_negative:minibatch_X_negative,
                                       mode:True})
                """
                print("Epoch:", epoch, "Minibatch:", i) 
                print("Positive score:")
                print(pos) 
                print("Negative score:")
                print(neg)
                print("ranking loss:", temp_cost)
                
                print("*************** End of a minibatch **********************************")
                """
                #print("Iteration ",i, ":  ",temp_cost)
                #minibatch_cost += temp_cost / num_minibatches

                """
                #print(tf.trainable_variables())
                
                variables_names = [v.name for v in tf.trainable_variables()]
                values = sess.run(variables_names)
                for k, v in zip(variables_names, values):
                    print("Variable: ", k)
                    print("Shape: ", v.shape)
                    #print(v)
                """

                if ((i+1) % opts.eval_minibatches) == 0:

                    # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    ########################Test Begins#####################################################
                    wins_count = 0
                    ties_count = 0
                    losses_count = 0

                    minibatches_test = mini_batches(X_test_1, X_test_0, opts.minibatch_size)

                    wins = tf.greater(score_positive, score_negative)
                    number_wins = tf.reduce_sum(tf.cast(wins, tf.int32))

                    ties = tf.equal(score_positive, score_negative)
                    number_ties = tf.reduce_sum(tf.cast(ties, tf.int32))

                    losses = tf.less(score_positive, score_negative)
                    number_losses = tf.reduce_sum(tf.cast(losses, tf.int32))

                    for (j, minibatch_test) in enumerate(minibatches_test):
                        (minibatch_X_positive, minibatch_X_negative) = minibatch_test

                        num_wins, num_ties, num_losses = sess.run([number_wins, number_ties, number_losses],
                                                                  feed_dict={X_positive: minibatch_X_positive,
                                                                             X_negative: minibatch_X_negative,
                                                                             mode: False})

                        wins_count += num_wins
                        ties_count += num_ties
                        losses_count += num_losses

                    recall = wins_count / (wins_count + ties_count + losses_count)

                    precision = wins_count / (wins_count + losses_count)

                    f1 = 2 * precision * recall / (precision + recall)

                    accuracy = wins_count / (wins_count + ties_count + losses_count)

                    # test_accuracy, test_f1 = sess.run([accuracy, f1], feed_dict={X_positive:X_test_1, X_negative:X_test_0})

                    # accuracy.eval(feed_dict={X_positive:X_test_1, X_negative:X_test_0})
                    # test_f1 = f1.eval({X_positive:X_test_1, X_negative:X_test_0})

                    print("\n\n")
                    print("***********Epoch: ", epoch, "    Minibatch: ", i, "  ******************")

                    print("Wins: ", wins_count)
                    print("Ties: ", ties_count)
                    print("losses: ", losses_count)

                    print(" -Test Accuracy:", accuracy)
                    print(" -Test F1 Score:", f1)

                    ########################Test Ends#####################################################
                    # """

            #print(minibatch_cost)
            #print("******************* End of an epoch ******************************")
            #print("******************* End of Training ******************************")

            # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            ########################Test Begins#####################################################
            wins_count = 0
            ties_count = 0
            losses_count = 0

            minibatches_test = mini_batches(X_test_1, X_test_0, opts.minibatch_size)

            wins = tf.greater(score_positive, score_negative)
            number_wins = tf.reduce_sum(tf.cast(wins, tf.int32))

            ties = tf.equal(score_positive, score_negative)
            number_ties = tf.reduce_sum(tf.cast(ties, tf.int32))

            losses = tf.less(score_positive, score_negative)
            number_losses = tf.reduce_sum(tf.cast(losses, tf.int32))

            for (j, minibatch_test) in enumerate(minibatches_test):
                (minibatch_X_positive, minibatch_X_negative) = minibatch_test

                num_wins, num_ties, num_losses = sess.run([number_wins, number_ties, number_losses],
                                                          feed_dict={X_positive: minibatch_X_positive,
                                                                     X_negative: minibatch_X_negative,
                                                                     mode: False})

                wins_count += num_wins
                ties_count += num_ties
                losses_count += num_losses

            recall = wins_count / (wins_count + ties_count + losses_count)

            precision = wins_count / (wins_count + losses_count)

            f1 = 2 * precision * recall / (precision + recall)

            accuracy = wins_count / (wins_count + ties_count + losses_count)

            # test_accuracy, test_f1 = sess.run([accuracy, f1], feed_dict={X_positive:X_test_1, X_negative:X_test_0})

            # accuracy.eval(feed_dict={X_positive:X_test_1, X_negative:X_test_0})
            # test_f1 = f1.eval({X_positive:X_test_1, X_negative:X_test_0})

            print("\n\n")
            print("***********Epoch: ", epoch, "    Minibatch: ", i, "  ******************")

            print("Wins: ", wins_count)
            print("Ties: ", ties_count)
            print("losses: ", losses_count)

            print(" -Test Accuracy:", accuracy)
            print(" -Test F1 Score:", f1)

            ########################Test Ends#####################################################
            # """

            #num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set


