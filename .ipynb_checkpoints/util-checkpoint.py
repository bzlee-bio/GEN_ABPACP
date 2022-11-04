import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import os
from Bio import SeqIO
from tensorflow.data import Dataset

class logHandler():
    def __init__(self, logdir):
        if not os.path.isdir('./w_and_l'):
            os.makedirs('./w_and_l')
        
        subPath = f'./w_and_l/{logdir}'
        i = 1 
        while True:
            subPath_temp = f'{subPath}_{i:02}'
            if not os.path.isdir(subPath_temp):
                os.makedirs(subPath_temp)
                subPath = subPath_temp
                break
            else:
                i+=1

        self.logPath = subPath
        self.weightPath = f'{subPath}/weights'
        os.makedirs(self.weightPath)

        self.tf_logger = tf.summary.create_file_writer(self.logPath) 
        self.logMessages = dict()
        self.bestModelInfo = [-1, 9999] # epoch, NLL loss

    def logStack(self, epoch, label, value):
        '''
        logging label types:
            G_Train/NLL_Summary
            G_Test/NLL_Summary
            G_Test/NLL_{each tgts}
            D_Train/Loss_{each tgts}
        '''
        if not label in self.logMessages.keys():
            self.logMessages[label] = list()
        self.logMessages[label]=[epoch, value]

    def write(self):
        flag = False
        with self.tf_logger.as_default():
            try:
                if self.logMessages['G_Test/NLL_Summary'][1]< self.bestModelInfo[1]: # Best performed model write
                    value = self.logMessages['G_Test/NLL_Summary'][1]
                    epoch = self.logMessages['G_Test/NLL_Summary'][0]
                    self.bestModelInfo = [epoch, value] # epoch, NLL loss
                    flag = True
            except KeyError:
                pass

            msgs = ''
            for label, l in sorted(self.logMessages.items()):
                epoch = l[0]
                value = l[1]
                tf.summary.scalar(label, value, step=epoch)
                if flag:
                    msgs = f'{msgs}<br>{label}, {value}'
                    tf.summary.scalar(f'Best/{label}', value, step=1)
            if flag:
                tf.summary.text('Best model',f'Epoch: {epoch}, {msgs}', step=0)

        self.logMessages = dict()
    


class DataHandler():
    def __init__(self, dataList_tr, dataList_test, params):
        self.maxSeqLen = params['maxSeqLen']
        self.batch_size = params['batch_size']
        self.keyToIdx = {k:i for i, k in enumerate(list(dataList_tr.keys()))}
        self.idxToKey = {i:k for k, i in self.keyToIdx.items()}
        self.vocab = {'a': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
                'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'p': 13, 'q': 14,
                'r': 15, 's': 16, 't': 17, 'v': 18, 'w': 19, 'y': 20,'<PAD>':0, '<SOS>':21}
        self.vocabInverse = {idx:tok for tok, idx in self.vocab.items()}
        self.tgts = list(dataList_tr.keys())
        self.seqLoader(dataList_tr, dataList_test)

    def fastaLoader(self, fileList, test=False):
        tot_data_biased = list()
        seq_len = list()
        batch_size_each = int(self.batch_size/len(self.tgts))
        tot_data = list()
        
        for _i, (tgt, fileName) in enumerate(fileList.items()):
            seqs = list()
            labels = list()
            for s in SeqIO.parse(fileName, 'fasta'):
                if len(s) <= self.maxSeqLen:
                    try:
                        seq = [self.vocab['<SOS>']] + [self.vocab[aa.lower()] for aa in s]
                        seqs.append(seq)
                        labels.append([self.keyToIdx[tgt]])
                    except KeyError:
                        print(f'{s} is removed from tgt {tgt}')
            seq_len.append(len(seqs))
            seqlen = len(seqs)
            print(f'Target: {tgt}, Sequence length: {seqlen}')
            seqs = tf.cast(self.padSequences(seqs), dtype=tf.int32)
            labels = tf.constant(labels, dtype=tf.int32)
            if not test:
                tot_data.append(Dataset.from_tensor_slices((seqs, labels)))         
                if _i==0:
                    biased_seqs = seqs
                    biased_labels = labels
                else:
                    biased_seqs = tf.concat([biased_seqs, seqs], axis=0)
                    biased_labels = tf.concat([biased_labels, labels], axis=0)
            else:
                tot_data.append(Dataset.from_tensor_slices((seqs, labels)))
        
        
        if not test:
            equal_data = list()
            biased_data = list()
            batch_size_biased = [int(self.batch_size*(x/sum(seq_len))) for x in seq_len]
            batch_size_biased[-1] = self.batch_size - sum(batch_size_biased[:-1])
            #print(batch_size_biased)
            for i, _data in enumerate(tot_data):
                equal_data.append(_data.shuffle(99999).batch(batch_size_each))
                #biased_data.append(_data.shuffle(99999).batch(batch_size_biased[i]))
            biased_data=Dataset.from_tensor_slices((biased_seqs, biased_labels)).shuffle(99999).batch(self.batch_size)
            return tf.data.Dataset.zip(tuple(equal_data)), biased_data
        else:
            return tf.data.Dataset.sample_from_datasets(tot_data)

    def padSequences(self, seq):
        return tf.keras.preprocessing.sequence.pad_sequences(sequences = seq, maxlen = self.maxSeqLen+1, padding='post')
        
    def seqLoader(self, dataList_tr, dataList_test):
        self.trDataset_equal, self.trDataset_biased  = self.fastaLoader(dataList_tr)
        self.testDataset = self.fastaLoader(dataList_test, test=True).batch(99999)
        
#        self.trDataset = balancedtrDataset.shuffle(buffer_size=99999).batch(self.batch_size)
       
#        trDatasets = Dataset.from_tensor_slices(())
#        self.trData = [tf.data.Dataset.from_tensor_slices(Data).shuffle(self.batch_size) for Data in trData.values()]
#        self.trDataLabel = 
#        self.testData = tf.data.Dataset.from_tensor_slices(testData).shuffle(self.batch_size)
#
#        weights = [1/len(self.tgts)]*len(self.tgts)
#        self.trDataSample = tf.data.Dataset.sample_from_datasets([dataset for dataset in self.trData.values()],
#                                                                 weights=weights)
#        
#        self.testDataSample = tf.data.Dataset.sample_from_datasets([dataset for dataset in self.testData.values()],
                                   #                              weights=weights)
#        print(trData.shape)
#        print(testData.shape)
