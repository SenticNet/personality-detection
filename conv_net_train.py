"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import os
import warnings
import sys
import time
import getpass
import csv
warnings.filterwarnings("ignore")

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def train_conv_net(datasets,
                   U,
                   ofile,
                   cv=0,
                   attr=0,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0][0])
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters

    #define model architecture
    index = T.lscalar()
    x = T.tensor3('x')
    y = T.ivector('y')
    mair = T.fmatrix('mair')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)

    conv_layers = []

    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, image_shape=None,
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        conv_layers.append(conv_layer)


    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],x.shape[1],x.shape[2],Words.shape[1]))

    def convolve_user_statuses(statuses):
        layer1_inputs = []

        def sum_mat(mat, out):
            z=ifelse(T.neq(T.sum(mat),T.constant(0)),T.constant(1),T.constant(0))
            return  out+z, theano.scan_module.until(T.eq(z,T.constant(0)))

        status_count,_ = theano.scan(fn = sum_mat, sequences=statuses, outputs_info=T.constant(0,dtype=theano.config.floatX))

        # Slice-out dummy (zeroed) sentences
        relv_input=statuses[:T.cast(status_count[-1],dtype='int32')].dimshuffle(0, 'x', 1, 2)

        for conv_layer in conv_layers:
            layer1_inputs.append(conv_layer.set_input(input=relv_input).flatten(2))

        features = T.concatenate(layer1_inputs, axis=1)

        avg_feat = T.max(features, axis=0)

        return avg_feat

    conv_feats, _ = theano.scan(fn= convolve_user_statuses, sequences= layer0_input)

    # Add Mairesse features
    layer1_input = T.concatenate([conv_feats, mair], axis=1)##mairesse_change
    hidden_units[0] = feature_maps*len(filter_hs) + datasets[4].shape[1]##mairesse_change
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    svm_data = T.concatenate([classifier.layers[0].output, y.dimshuffle(0, 'x')], axis = 1)
    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        rand_perm = np.random.permutation(range(len(datasets[0])))
        train_set_x = datasets[0][rand_perm]
        train_set_y = datasets[1][rand_perm]
        train_set_m = datasets[4][rand_perm]
        extra_data_x = train_set_x[:extra_data_num]
        extra_data_y = train_set_y[:extra_data_num]
        extra_data_m = train_set_m[:extra_data_num]
        new_data_x = np.append(datasets[0],extra_data_x,axis=0)
        new_data_y = np.append(datasets[1],extra_data_y,axis=0)
        new_data_m = np.append(datasets[4],extra_data_m,axis=0)
    else:
        new_data_x = datasets[0]
        new_data_y = datasets[1]
        new_data_m = datasets[4]
    rand_perm = np.random.permutation(range(len(new_data_x)))
    new_data_x = new_data_x[rand_perm]
    new_data_y = new_data_y[rand_perm]
    new_data_m = new_data_m[rand_perm]
    n_batches = new_data_x.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets
    test_set_x = datasets[2]
    test_set_y = np.asarray(datasets[3],"int32")
    test_set_m = datasets[5]
    train_set_x, train_set_y, train_set_m = shared_dataset((new_data_x[:n_train_batches*batch_size], new_data_y[:n_train_batches*batch_size], new_data_m[:n_train_batches*batch_size]))
    val_set_x, val_set_y, val_set_m = shared_dataset((new_data_x[n_train_batches*batch_size:], new_data_y[n_train_batches*batch_size:], new_data_m[n_train_batches*batch_size:]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size],
              mair: val_set_m[index * batch_size: (index + 1) * batch_size]},##mairesse_change
                                allow_input_downcast = True)

    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], [classifier.errors(y), svm_data],
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size],
                  mair: train_set_m[index * batch_size: (index + 1) * batch_size]},##mairesse_change
                                 allow_input_downcast=True)
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size],
                mair: train_set_m[index * batch_size: (index + 1) * batch_size]},##mairesse_change
                                  allow_input_downcast = True)

    test_y_pred = classifier.predict(layer1_input)
    test_error = T.sum(T.neq(test_y_pred, y))
    true_p = T.sum(test_y_pred*y)
    false_p = T.sum(test_y_pred*T.mod(y+T.ones_like(y),T.constant(2,dtype='int32')))
    false_n = T.sum(y*T.mod(test_y_pred+T.ones_like(y),T.constant(2,dtype='int32')))
    test_model_all = theano.function([x, y,
                                        mair##mairesse_change
                                        ]
                                    , [test_error, true_p, false_p, false_n, svm_data], allow_input_downcast = True)

    test_batches = test_set_x.shape[0]/batch_size;


    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    fscore = 0
    cost_epoch = 0
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean([loss[0] for loss in train_losses])
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        epoch_perf = 'epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.)
        print(epoch_perf)
        ofile.write(epoch_perf+"\n")
        ofile.flush()
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss_list = [test_model_all(test_set_x[idx*batch_size:(idx+1)*batch_size], test_set_y[idx*batch_size:(idx+1)*batch_size],
            test_set_m[idx*batch_size:(idx+1)*batch_size]##mairesse_change
            ) for idx in xrange(test_batches)]
            if test_set_x.shape[0]>test_batches*batch_size:
                test_loss_list.append(test_model_all(test_set_x[test_batches*batch_size:], test_set_y[test_batches*batch_size:],
                test_set_m[test_batches*batch_size:]##mairesse_change
                ))
            test_loss_list_temp=test_loss_list
            test_loss_list=np.asarray([t[:-1] for t in test_loss_list])
            test_loss = np.sum(test_loss_list[:, 0])/float(test_set_x.shape[0])
            test_perf = 1- test_loss
            tp = np.sum(test_loss_list[:, 1])
            fp = np.sum(test_loss_list[:, 2])
            fn = np.sum(test_loss_list[:, 3])
            tn = test_set_x.shape[0]-(tp+fp+fn)
            fscore=np.mean([2*tp/float(2*tp+fp+fn), 2*tn/float(2*tn+fp+fn)])
            svm_test=np.concatenate([t[-1] for t in test_loss_list_temp], axis=0)
            svm_train=np.concatenate([t[1] for t in train_losses], axis=0)
            output="Test result: accu: "+str(test_perf)+", macro_fscore: "+str(fscore)+"\ntp: "+str(tp)+" tn:"+str(tn)+" fp: "+str(fp)+" fn: "+str(fn)
            print output
            ofile.write(output+"\n")
            ofile.flush()
            # dump train and test features
            cPickle.dump(svm_test, open("cvte"+str(attr)+str(cv)+".p", "wb"))
            cPickle.dump(svm_train, open("cvtr"+str(attr)+str(cv)+".p", "wb"))
        updated_epochs = refresh_epochs()
        if updated_epochs!=None and n_epochs!=updated_epochs:
            n_epochs = updated_epochs
            print 'Epochs updated to '+str(n_epochs)
    return test_perf, fscore

def refresh_epochs():
    try:
        f=open('n_epochs','r')
    except Exception:
        return None

    try:
        n = int(f.readline().strip())
    except Exception:
        f.close()
        return None
    f.close()
    return n


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y, data_m = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_m = theano.shared(np.asarray(data_m,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32'), shared_m

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

def get_idx_from_sent(status, word_idx_map, charged_words, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    length = len(status)


    pass_one=True
    while len(x)==0:
        for i in range(length):
            words = status[i].split()
            if pass_one:
                words_set = set(words)
                if len(charged_words.intersection(words_set))==0:
                    continue
            else:
                if np.random.randint(0,2)==0:
                    continue
            y=[]
            for i in xrange(pad):
                y.append(0)
            for word in words:
                if word in word_idx_map:
                    y.append(word_idx_map[word])

            while len(y) < max_l+2*pad:
                y.append(0)
            x.append(y)
        pass_one=False

    if len(x) < max_s:
        x.extend([[0]*(max_l+2*pad)]*(max_s-len(x)))



    return x

def make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, cv, per_attr=0, max_l=51, max_s=200, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, testX, trainY, testY, mTrain, mTest = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map,
        charged_words,
        max_l, max_s, k, filter_h)

        if rev["split"]==cv:
            testX.append(sent)
            testY.append(rev['y'+str(per_attr)])
            mTest.append(mairesse[rev["user"]])
        else:
            trainX.append(sent)
            trainY.append(rev['y'+str(per_attr)])
            mTrain.append(mairesse[rev["user"]])
    trainX = np.array(trainX,dtype="int32")
    testX = np.array(testX,dtype="int32")
    trainY = np.array(trainY,dtype="int32")
    testY = np.array(testY,dtype="int32")
    mTrain = np.array(mTrain, dtype=theano.config.floatX)
    mTest = np.array(mTest, dtype=theano.config.floatX)
    return [trainX, trainY, testX, testY, mTrain, mTest]


if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("essays_mairesse.p","rb"))
    revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    attr = int(sys.argv[3])
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W

    r = range(0,10)

    ofile=file('perf_output_'+str(attr)+'.txt','w')

    charged_words=[]

    emof=open("Emotion_Lexicon.csv","rb")
    csvf=csv.reader(emof, delimiter=',',quotechar='"')
    first_line=True

    for line in csvf:
        if first_line:
            first_line=False
            continue
        if line[11]=="1":
            charged_words.append(line[0])

    emof.close()

    charged_words=set(charged_words)

    results = []
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, i, attr, max_l=149, max_s=312, k=300, filter_h=3)

        perf, fscore = train_conv_net(datasets,
                              U,
                              ofile,
                              cv=i,
                              attr=attr,
                              lr_decay=0.95,
                              filter_hs=[1,2,3],
                              conv_non_linear="relu",
                              hidden_units=[200,200,2],
                              shuffle_batch=True,
                              n_epochs=50,
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5, 0.5, 0.5],
                              activations=[Sigmoid])
        output = "cv: " + str(i) + ", perf: " + str(perf)+ ", macro_fscore: " + str(fscore)
        print output
        ofile.write(output+"\n")
        ofile.flush()
        results.append([perf, fscore])
    results=np.asarray(results)
    perf_out = 'Perf : '+str(np.mean(results[:, 0]))
    fscore_out = 'Macro_Fscore : '+str(np.mean(results[:, 1]))
    print perf_out
    print fscore_out
    ofile.write(perf_out+"\n"+fscore_out)
    ofile.close()
