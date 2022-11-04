import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle


class Generator(keras.Model):
    def __init__(self, params, DataHandler):
        super(Generator, self).__init__()
        self.Data = DataHandler
        self.batch_size = params['batch_size']
        self.vocab_size = len(self.Data.vocab)
        self.tgt_len = len(self.Data.tgts)
        self.maxSeqLen = params['maxSeqLen']
        ### GEN module
        self.rnn_based = list()
        for i, unit in enumerate(params['G_units']):
            name = f'rnn_based {i}'
            self.rnn_based.append(layers.GRU(unit, return_state=True, return_sequences=True, name=name)) 
        self.embedding = layers.Embedding(self.vocab_size, params['embed_dim'], name='Embed')
        self.dense = layers.Dense(self.vocab_size-1, name='Dense')
        self.embedding_tgt = layers.Embedding(self.tgt_len,5)
    def probController(self, inp):
         pad_idx = tf.where(tf.math.equal(inp, self.Data.vocab['<PAD>']))
         pad_vocab_num = tf.broadcast_to([self.Data.vocab['<PAD>']],shape=[pad_idx.shape[0],1])
 
         ### Data type cast
         pad_idx = tf.cast(pad_idx,dtype=tf.int64)
         pad_vocab_num = tf.cast(pad_vocab_num, dtype=tf.int64)
 
         pad_idx = tf.concat([pad_idx, pad_vocab_num], axis=1)
         dense_shape = list(inp.shape)+[self.vocab_size-1]
         pad_sparse = tf.SparseTensor(indices = pad_idx, values=[99.]*len(pad_idx), dense_shape=dense_shape)
         pad_prob = tf.sparse.to_dense(pad_sparse)
 
         return pad_prob
    

    def call(self, inputs, labels, states=None, training=False):
        ### input shape: batch_size, seq_len   labels shape: batch_size, 1
        x = self.embedding(inputs, training = training) # batch_size, input_len, embed_dim
        x_lab = tf.repeat(labels, repeats=x.shape[1], axis=-1) # batch_size, input_len
        #x_lab = self.embedding_tgt(x_lab, training = training)
        x_lab = tf.one_hot(x_lab, self.tgt_len) # batch_size, input_len, tgt_len
        x = layers.concatenate([x, x_lab], -1)

        x_mask = tf.cast(tf.math.not_equal(inputs, self.Data.vocab['<PAD>']), tf.float32)
        newStates = list()
        for i, rnn_layers in enumerate(self.rnn_based):
            if states is not None:
                st = states[i]
            else:
                st = None
            
            x, state = rnn_layers(x, initial_state = st, training=training)
           # x = layers.concatenate([x, x_lab], -1)
            newStates.append(state)

        #x = layers.concatenate([x, x_lab], -1)
        x = self.dense(x, training=training) # batch_size, input_len, vocab_size
        logit = x + self.probController(inputs)
        #print(x, logit)
        return logit, newStates, x_mask

    def genSeqs(self, inp = None, training=False, seqLen=None, states=None, batch_size = None, label=None, return_label = False):
        if seqLen is None:
            seqLen = self.maxSeqLen
        
        
        if batch_size is None:
            batch_size = self.batch_size

        if label is None:
            label_each_len = int(batch_size/self.tgt_len)
            assert batch_size%self.tgt_len==0, 'batch size should be provided as multiple of number of targets'
            label = tf.constant([], dtype=tf.int32)
            for tgt_idx in range(self.tgt_len):
                label = tf.concat([label, tf.repeat(tgt_idx, repeats=label_each_len)], axis=-1)
        label = tf.reshape(label, (-1, 1))

        tot_gen_prob = tf.TensorArray(tf.float32, size=seqLen, clear_after_read=True)
        tot_gen_seqs = tf.TensorArray(tf.int32, size=seqLen, clear_after_read=True)
        tot_gen_mask = tf.TensorArray(tf.float32, size=seqLen, clear_after_read=True)

        arange_tensor = tf.reshape(tf.range(0,batch_size, dtype=tf.int32), (-1,1))
        ### Initial input configuration
        if inp is None:
            tot_states = [None]
            inputs = tf.fill([batch_size, 1], self.Data.vocab['<SOS>'])
            start_token = inputs
        else:
            inputs = inp
            tot_states = list()
            #inp = tf.constant([self.Data.vocab['<SOS>']]*batch_size, dtype=tf.float32)
        for i in range(seqLen):
            logit, states, mask = self.call(inputs = inputs, labels = label, states = states, training = training)
            gen_prob = tf.cast(tf.nn.softmax(logit, axis=-1), dtype=tf.float32)

            sampled_idx = self.sampling_from_logits(logit)
            sampled_idx_arange = tf.concat([arange_tensor, sampled_idx], axis=-1)
            tot_gen_prob = tot_gen_prob.write(i, tf.squeeze(tf.gather_nd(tf.squeeze(gen_prob),
                                                              tf.squeeze(sampled_idx_arange))))
            tot_gen_seqs = tot_gen_seqs.write(i, tf.squeeze(sampled_idx))
            tot_gen_mask = tot_gen_mask.write(i, tf.squeeze(mask))

            inputs = sampled_idx
            tot_states.append(states)
            # print(gen_prob, tf.squeeze(sampled_idx), tf.squeeze(tf.gather_nd(tf.squeeze(gen_prob),
                                                              # tf.squeeze(sampled_idx_arange))))
        tot_gen_prob = tf.transpose(tot_gen_prob.stack())
        tot_gen_seqs = tf.transpose(tot_gen_seqs.stack())
        tot_gen_mask = tf.transpose(tot_gen_mask.stack())
        if inp is None:
            tot_gen_prob = tf.concat([tf.ones([tot_gen_prob.shape[0], 1], dtype=tf.float32), tot_gen_prob], axis=1)
            tot_gen_seqs = tf.concat([start_token, tot_gen_seqs], axis=1)
            tot_gen_mask = tf.concat([tf.ones([tot_gen_prob.shape[0], 1], dtype=tf.float32), tot_gen_mask], axis=1)

        if return_label:
            return tot_gen_seqs, tot_gen_prob, tot_gen_mask, tot_states, label
        return tot_gen_seqs, tot_gen_prob, tot_gen_mask, tot_states



    def sampling_from_logits(self, x):
        return tf.random.categorical(tf.squeeze(x), 1, dtype=tf.int32)
            
    def save_model_weight(self, path, epoch):
        fName = f'{path}/G_weight_{epoch}.weight'
        
        with open(fName, 'wb') as f:
            pickle.dump(self.get_weights(), f)
        
    def load_model_weight(self, path, epoch):
        fName = f'{path}/weights/G_weight_{epoch}.weight'
        with open(fName, 'rb') as f:
            weights = pickle.load(f)
            self.set_weights(weights)
        return fName











