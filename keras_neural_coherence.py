from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Flatten, Dense, Dropout, merge, concatenate
# from keras.layers import concatenate
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
import numpy as np
from utilities import my_callbacks
from utilities import data_helper
import optparse
import sys

np.set_printoptions(threshold=np.nan)


def ranking_loss(y_true, y_pred):
    pos = y_pred[:, 0]
    neg = y_pred[:, 1]
    # loss = -K.sigmoid(pos-neg) # use
    loss = K.maximum(1.0 + neg - pos, 0.0)  # if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true


if __name__ == '__main__':
    # parse user input
    print("Starting...1:42")
    parser = optparse.OptionParser("%prog [options]")

    #file related options
    parser.add_option("-g", "--log-file",   dest="log_file", help="log file [default: %default]")
    parser.add_option("-d", "--data-dir",   dest="data_dir", help="directory containing list of train, test and dev file [default: %default]")
    parser.add_option("-m", "--model-dir",  dest="model_dir", help="directory to save the best models [default: %default]")

    parser.add_option("-t", "--max-length", dest="maxlen", type="int", help="maximul length (for fixed size input) [default: %default]") # input size
    parser.add_option("-f", "--nb_filter",         dest="nb_filter",     type="int",   help="nb of filter to be applied in convolution over words [default: %default]")
    #parser.add_option("-r", "--filter_length",     dest="filter_length", type="int",   help="length of neighborhood in words [default: %default]")
    parser.add_option("-w", "--w_size",         dest="w_size", type="int",   help="window size length of neighborhood in words [default: %default]")
    parser.add_option("-p", "--pool_length",       dest="pool_length",   type="int",   help="length for max pooling [default: %default]")
    parser.add_option("-e", "--emb-size",          dest="emb_size",      type="int",   help="dimension of embedding [default: %default]")
    parser.add_option("-s", "--hidden-size",       dest="hidden_size",   type="int",   help="hidden layer size [default: %default]")
    parser.add_option("-o", "--dropout_ratio",     dest="dropout_ratio", type="float", help="ratio of cells to drop out [default: %default]")

    parser.add_option("-a", "--learning-algorithm", dest="learn_alg", help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %default]")
    parser.add_option("-b", "--minibatch-size",     dest="minibatch_size", type="int", help="minibatch size [default: %default]")
    parser.add_option("-l", "--loss",               dest="loss", help="loss type (hinge, squared_hinge, binary_crossentropy) [default: %default]")
    parser.add_option("-n", "--epochs",             dest="epochs", type="int", help="nb of epochs [default: %default]")
    parser.add_option("-P", "--permutation",        dest="p_num", type="int", help="nb of permutation[default: %default]")
    parser.add_option("-F", "--feats",        dest="f_list", help="semantic features using in the model, separate by . [default: %default]")

    parser.set_defaults(

        data_dir        = "./data/"
        ,log_file       = "log"
        ,model_dir      = "./saved_models/"

        ,learn_alg      = "rmsprop" # sgd, adagrad, rmsprop, adadelta, adam (default)
        ,loss           = "ranking_loss" # hinge, squared_hinge, binary_crossentropy (default)
        ,minibatch_size = 32
        ,dropout_ratio  = 0.5

        ,maxlen         = 14000
        ,epochs         = 15
        ,emb_size       = 100
        ,hidden_size    = 250
        ,nb_filter      = 150
        ,w_size         = 6
        ,pool_length    = 6
        ,p_num          = 20
        ,f_list         = "" #"0.1.3"
    )

    opts, args = parser.parse_args(sys.argv)


    print('Loading vocab of the whole dataset...')
    vocab = data_helper.load_all(filelist="data/wsj.all")
    print(vocab)

    print("loading entity-grid for pos and neg documents...")
    X_train_1, X_train_0, E = data_helper.load_and_numberize_Egrid_with_Feats("data/wsj.train",
                                                                              perm_num=opts.p_num, maxlen=opts.maxlen,
                                                                              window_size=opts.w_size, vocab_list=vocab,
                                                                              emb_size=opts.emb_size)

    X_dev_1, X_dev_0, E = data_helper.load_and_numberize_Egrid_with_Feats("data/wsj.dev",
                                                                          perm_num=opts.p_num, maxlen=opts.maxlen,
                                                                          window_size=opts.w_size, E=E,
                                                                          vocab_list=vocab, emb_size=opts.emb_size)

    X_test_1, X_test_0, E = data_helper.load_and_numberize_Egrid_with_Feats("data/wsj.test",
                                                                            perm_num=opts.p_num, maxlen=opts.maxlen,
                                                                            window_size=opts.w_size, E=E,
                                                                            vocab_list=vocab, emb_size=opts.emb_size)

    num_train = len(X_train_1)
    num_dev = len(X_dev_1)
    num_test = len(X_test_1)
    # assign Y value
    y_train_1 = [1] * num_train
    y_dev_1 = [1] * num_dev
    y_test_1 = [1] * num_test

    print('.....................................')
    print("Num of traing pairs: " + str(num_train))
    print("Num of dev pairs: " + str(num_dev))
    print("Num of test pairs: " + str(num_test))
    # print("Num of permutation in train: " + str(opts.p_num))
    # print("The maximum in length for CNN: " + str(opts.maxlen))
    print('.....................................')



    # One hot encoding of the outputs
    y_train_1 = np_utils.to_categorical(y_train_1, 2)
    y_dev_1 = np_utils.to_categorical(y_dev_1, 2)
    y_test_1 = np_utils.to_categorical(y_test_1, 2)

    # randomly shuffle the training data
    np.random.seed(113)
    np.random.shuffle(X_train_1)
    np.random.seed(113)
    np.random.shuffle(X_train_0)


    # first, define a CNN model for sequence of entities
    sent_input = Input(shape=(opts.maxlen,), dtype='int32', name='sent_input')

    # embedding layer encodes the input into sequences of 300-dimenstional vectors.

    E = np.float32(E)  # E was float64 which doesn't work for tensorflow conv1d function
    x = Embedding(input_dim=len(vocab), output_dim=opts.emb_size, weights=[E], input_length=opts.maxlen)(sent_input)

    # add a convolutiaon 1D layer
    # x = Dropout(dropout_ratio)(x)
    filter_init = tf.keras.initializers.glorot_uniform(seed=2018)
    x = Convolution1D(filters=opts.nb_filter, kernel_size=opts.w_size, padding='valid', activation='relu', kernel_initializer=filter_init)(x)

    # add max pooling layers
    # x = AveragePooling1D(pool_length=pool_length)(x)
    x = MaxPooling1D(pool_size=opts.pool_length)(x)

    # x = Dropout(opts.dropout_ratio)(x)
    x = Flatten()(x)

    # x = Dense(hidden_size, activation='relu')(x)
    x = Dropout(opts.dropout_ratio, seed=2018)(x)

    # add latent coherence score
    v_init = tf.keras.initializers.glorot_uniform(seed=2018)
    out_x = Dense(1, activation=None, kernel_initializer=v_init)(x)

    shared_cnn = Model(sent_input, out_x)



    print(shared_cnn.summary())



    # Inputs of pos and neg document
    pos_input = Input(shape=(opts.maxlen,), dtype='int32', name="pos_input")
    neg_input = Input(shape=(opts.maxlen,), dtype='int32', name="neg_input")

    # these two models will share eveything from shared_cnn
    pos_branch = shared_cnn(pos_input)
    neg_branch = shared_cnn(neg_input)

    concatenated = merge([pos_branch, neg_branch], mode='concat', name="coherence_out")
    # concatenated = concatenate([pos_branch, neg_branch], name="coherence_out")
    # output is two latent coherence score


    final_model = Model([pos_input, neg_input], concatenated)

    # final_model.compile(loss='ranking_loss', optimizer='adam')
    final_model.compile(loss={'coherence_out': ranking_loss}, optimizer="rmsprop")

    # setting callback
    histories = my_callbacks.Histories()

    # print(shared_cnn.summary())






    for ep in range(opts.epochs):

        # Train Phase:

        final_model.fit([X_train_1, X_train_0], y_train_1, validation_data=None, shuffle=False, epochs=1,
                        verbose=0, batch_size=32, callbacks=[histories])
        """
        for (i, loss) in enumerate(histories.losses):
            print("Iteration ", i, ":  ", loss)

        """

        #"""
        # Test Phase:
    
        y_pred = final_model.predict([X_test_1, X_test_0])
    
        ties = 0
        wins = 0
        n = len(y_pred)
        for i in range(0, n):
            if y_pred[i][0] > y_pred[i][1]:
                wins = wins + 1
            elif y_pred[i][0] == y_pred[i][1]:
                ties = ties + 1
        # print("Perform on test set after Epoch: " + str(ep) + "...!")
        # print(" -Wins: " + str(wins) + " Ties: "  + str(ties))
        loss = n - (wins + ties)
    
        recall = wins / n;
        prec = wins / (wins + loss)
        f1 = 2 * prec * recall / (prec + recall)

        print("\n\n")
        print("***********Epoch: ", ep, "  ******************")
    
        print("Wins: ", wins)
        print("Ties: ", ties)
        print("losses: ", loss)
    
        print(" -Test Accuracy: " + str(wins / n))
        print(" -Test F1 Score: " + str(f1))

        #"""