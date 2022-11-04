import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

class Discriminator(keras.Model):
    def __init__(self, params, DataHandler):
        super(Discriminator, self).__init__()
        self.Data = DataHandler
        self.batch_size = params['batch_size']
        self.vocab_size = len(self.Data.vocab)
        self.tgt_len = len(self.Data.tgts)
        self.rnn_based = list()
        
        self.layers_tgt = dict()
        for tgt in self.Data.tgts:
            self.layers_tgt[tgt] = [layers.Embedding(self.vocab_size, params['embed_dim'], name=f'Embed_{tgt}', mask_zero = True)]
            self.layers_tgt[tgt].append(layers.GRU(100, name=f'Disc_RNN_1_{tgt}', return_sequences=True))
            self.layers_tgt[tgt].append(layers.GRU(100, name=f'Disc_RNN_2_{tgt}'))
            self.layers_tgt[tgt].append(layers.Dense(1, name=f'Disc_FC_{tgt}'))
#        for i, unit in enumerate([50]):
#            self.rnn_based.append(layers.GRU(unit, name=f'Disc_RNN_{i}'))
#        self.FC = [#layers.Dense(10, name='Disc_FC1'),
#                    layers.Dense(1, name='Disc_FC2')]
        
        
    def call(self, inp, label, tgt, training=False):
        x = inp
        for forward in self.layers_tgt[tgt]:
            x = forward(x, training=training)

        return x
#        x = self.embedding(inp, training=training)
##        x_lab = tf.repeat(label, repeats=x.shape[1], axis=-1)
##        x_lab = tf.one_hot(x_lab, self.tgt_len)
##        x = layers.concatenate([x, x_lab], -1)
#
#        for i, rnn_layer in enumerate(self.rnn_based):
#            x = rnn_layer(x, training=training)
#
#        for i, FC in enumerate(self.FC):
#            x = FC(x, training=training)
#
#        return x

    def init_layer(self):
        inp = tf.fill([self.batch_size, 2], 0)
        label = None
        for tgt in self.Data.tgts:
            self.call(inp = inp, label = label, tgt = tgt)

    def save_model_weights(self, path, epoch):
        fName = f'{path}/D_weight_{epoch}.weight'
        print(len(self.get_weights()))
        with open(fName, 'wb') as f:
            pickle.dump(self.get_weights(), f)

    def load_model_weight(self, path, epoch):
        fName = f'{path}/weights/D_weight_{epoch}.weight'

        with open(fName, 'rb') as f:
            weights = pickle.load(f)
            self.set_weights(weights)
        return fName

    def return_targeted_trainable_variables(self, tgt):
        return [var for var in self.trainable_variables if tgt in var.name]
