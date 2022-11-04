import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers

class modelTrainer(object):
    def __init__(self, params, Data, G, D):
        self.G = G
        self.D = D
        self.Data = Data
        self.tgts = self.Data.tgts
        self.batch_size = params['batch_size']
        self.maxSeqLen = params['maxSeqLen']
        # self.ro_iter = params['rollout_iter'] # num of rollout iter

        self.G_pre_loss = losses.SparseCategoricalCrossentropy(from_logits = True, 
                                                                reduction = losses.Reduction.NONE)
        self.D_loss = losses.BinaryCrossentropy(from_logits=True)

        if params['G_pre_lr'] is not None:
            self.G_pre_optim = optimizers.Adam(learning_rate = params['G_pre_lr'])
        # if params['D_pre_lr'] is not None:
        #     self.D_pre_optim = optimizers.Adam(learning_rate = params['D_pre_lr'])
        # if params['G_lr'] is not None:
        #     self.G_optim = optimizers.Adam(learning_rate = params['G_lr'])
        # if params['D_lr'] is not None:
        #     self.D_optim = optimizers.Adam(learning_rate = params['D_lr'])

    def G_NLL_test_loss(self):
        loss = dict()
        loss_summary = list()
        for predTgtIdx, predTgt in enumerate(self.tgts):
            x_batch, label = list(self.Data.testDataset.as_numpy_iterator())[0]
            tgtLabel = tf.ones_like(label, dtype=tf.int32)*predTgtIdx
            losses = self.G_NLL_loss(x_batch = x_batch, label = tgtLabel, training = False, test=True)
            for i, originalTgt in enumerate(self.tgts):
                key = f'Data_{originalTgt}_Model_{predTgt}'
                idx = tf.math.equal(label, i)
                selected_losses = losses[idx]
                #print(key, selected_losses)
                loss[key]= tf.math.reduce_mean(selected_losses)
                if predTgt == originalTgt:
                    loss_summary.append(loss[key])
        loss['Summary'] = tf.math.reduce_mean(loss_summary)
        return loss


    def G_NLL_loss(self, x_batch, label, training = False, test = False):
        x_inp = x_batch[:,:-1]
        true = x_batch[:,1:]

        logit, _, mask = self.G(inputs = x_inp, labels = label, training = training)
        loss_no_reduced = self.G_pre_loss(true, logit)
        row_mask_num = tf.math.reduce_sum(mask, 1, keepdims = True)
        loss_masked = tf.math.multiply(loss_no_reduced, mask)
        loss_divided = tf.math.divide(tf.math.reduce_sum(loss_masked, 1,keepdims=True), row_mask_num)

        if test:
            return loss_divided
        else:
            weight = np.ones_like(label.numpy(), dtype=float)
            for tgtIdx, tgt in enumerate(self.tgts):
                idxs = tf.math.equal(label,tf.constant(tgtIdx, dtype=tf.int32))
                weight[tf.squeeze(tf.where(idxs)[:,0]),:]/= tf.math.reduce_sum(tf.cast(idxs,dtype=tf.float32)).numpy()
            loss = tf.reduce_sum(tf.math.multiply(loss_divided, weight))/len(self.tgts)

        return loss

    def G_pre_train(self, lossCheck = False):
        losses = tf.constant([])
        # for i, _data in enumerate(self.Data.trDataset_biased):
            # ### Stack batch dataset from multiple targets
            # for j, (x_temp, label_temp) in enumerate(_data):
            #     if j==0:
            #         x_batch = x_temp
            #         label = label_temp
            #     else:
            #         x_batch = tf.concat([x_batch, x_temp], axis=0)
            #         label = tf.concat([label, label_temp], axis=0)
            #     #print(label, tf.math.reduce_sum(label))
        for i, (x_batch, label) in enumerate(self.Data.trDataset_biased):            
            with tf.GradientTape() as tape:
                loss = self.G_NLL_loss(x_batch = x_batch, label = label, training=True)
            if not lossCheck:
                grad = tape.gradient(loss, self.G.trainable_variables)
                self.G_pre_optim.apply_gradients(zip(grad, self.G.trainable_variables))
            losses = tf.concat([losses, tf.reshape(loss, (-1))], axis=-1)
        return tf.math.reduce_mean(losses)
    
    #@tf.function
    def D_grad_update(self, x, label, y,tgt,  pretrain = False):
        with tf.GradientTape() as tape:
            logit = self.D(inp = x, label = label, tgt=tgt, training=True)
            loss = self.D_loss(y, logit)
        D_tgt_tr_var = self.D.return_targeted_trainable_variables(tgt)
        grad = tape.gradient(loss, D_tgt_tr_var)
        if pretrain:
            self.D_pre_optim.apply_gradients(zip(grad, D_tgt_tr_var))
        else:
            self.D_optim.apply_gradients(zip(grad, D_tgt_tr_var))
        return loss

    def D_pre_train(self, spTgt = None):
        sub_iter = 3
        losses = list()
        if spTgt is None:
            tgts = self.Data.tgts
        else:
            tgts = [spTgt]
        for i, _data in enumerate(self.Data.trDataset_biased):
            for tgtIdx, tgt in enumerate(tgts):
            # for curr_tgtidx, (x_batch, label_batch) in enumerate(_data):
                tgt_data = tf.math.equal(_data[1], tgtIdx)
                # print(tgt_data)
                x_batch = tf.boolean_mask(_data[0],tf.squeeze(tgt_data), 0)
                label_batch = tf.boolean_mask(_data[1],tf.squeeze(tgt_data), 0)
                true_len = x_batch.shape[0]
                #print(x_batch.shape, label_batch.shape)
                #lab_input = tf.repeat(tgt_idx, repeats=true_len)
                fake_seqs, _, _, _, fake_label = self.G.genSeqs(label = label_batch, batch_size = true_len, return_label=True)
                x_tot = tf.concat([x_batch, fake_seqs], 0)
                lab_tot = tf.concat([label_batch, fake_label], 0)
                y_tot = tf.concat([tf.ones([true_len, 1]), tf.zeros([true_len, 1])],0)

                for k in range(sub_iter):
                    loss = self.D_grad_update(x_tot, lab_tot, y_tot, self.Data.idxToKey[tgtIdx], pretrain=True)
                    losses.append(loss)
        return tf.math.reduce_mean(losses)

    
    def D_train(self, spTgt = None):
        losses = list()
        for i, _data in enumerate(self.Data.trDataset_equal):
            for tgtIdx, (x_batch, label_batch) in enumerate(_data):
                if spTgt == self.Data.idxToKey[tgtIdx] or spTgt is None:
                    # tgt_data = tf.math.equal(_data[1], tgtIdx)
                    # # print(tgt_data)
                    # x_batch = tf.boolean_mask(_data[0],tf.squeeze(tgt_data), 0)
                    # label_batch = tf.boolean_mask(_data[1],tf.squeeze(tgt_data), 0)
                    true_len = x_batch.shape[0]
                    #print(x_batch.shape, label_batch.shape)
                    #lab_input = tf.repeat(tgt_idx, repeats=true_len)
                    fake_seqs, _, _, _, fake_label = self.G.genSeqs(label = label_batch, batch_size = true_len, return_label=True)
                    x_tot = tf.concat([x_batch, fake_seqs], 0)
                    lab_tot = tf.concat([label_batch, fake_label], 0)
                    y_tot = tf.concat([tf.ones([true_len, 1]), tf.zeros([true_len, 1])],0)

                    for k in range(3):
                        loss = self.D_grad_update(x_tot, lab_tot, y_tot, self.Data.idxToKey[tgtIdx], pretrain=False)
                        losses.append(loss)
        return tf.math.reduce_mean(losses)    

    
    
    def rollout(self, spTgt = None):
        if spTgt is None:
            gen_seqs, gen_prob, gen_mask, tot_states, label = self.G.genSeqs(return_label=True)
            _tgt = self.Data.tgts
        else:
            label = tf.fill([self.batch_size, 1],self.Data.keyToIdx[spTgt])
            gen_seqs, gen_prob, gen_mask, tot_states = self.G.genSeqs(label=label)
            _tgt = [spTgt]
        #print(label, gen_seqs, gen_prob, gen_mask)
        disc_prob = np.zeros((self.batch_size, self.maxSeqLen+1))
        
        for i in range(0, self.maxSeqLen+1):
            gen_seq_len = self.maxSeqLen-i
            inp_seq = tf.reshape(gen_seqs[:,i],(-1,1))
            for ro_iter in range(self.ro_iter):
                if gen_seq_len !=0:
                    ro_seq, _, _, _ = self.G.genSeqs(inp=inp_seq, training=True, seqLen = gen_seq_len, states=tot_states[i], label=label)
                    ro_tot_seq = tf.concat([gen_seqs[:,:i+1], ro_seq], axis=-1)                    
                else:
                    ro_tot_seq = gen_seqs
                    
               # print(ro_tot_seq.shape)
                disc_logit = None
                for tgt in _tgt:
                    tgtidx = self.Data.keyToIdx[tgt]
                    #print('TGTIDX --------------------', tgtidx)
                    ro_temp = tf.boolean_mask(ro_tot_seq, tf.squeeze(tf.math.equal(label, tgtidx)),0)
                    label_temp = tf.boolean_mask(label, tf.squeeze(tf.math.equal(label, tgtidx)),0)
                    #print(ro_temp, label_temp, tgt)
                    logit_temp = self.D(inp = ro_temp, label=label_temp, tgt=tgt, training=False)
                    if disc_logit is None:
                        disc_logit = logit_temp
                    else:
                        disc_logit = tf.concat([disc_logit, logit_temp], 0)
                if gen_seq_len!=0:
                    disc_prob[:,i] += tf.squeeze(keras.activations.sigmoid(disc_logit)).numpy()
                else:
                    disc_prob[:,i] += tf.squeeze(keras.activations.sigmoid(disc_logit)).numpy()*self.ro_iter
                    break
        disc_prob /= self.ro_iter                
        print(gen_seqs)
        print(disc_prob)
#        print(disc_prob.shape)
        #row_mask_num = tf.math.reduce_sum(gen_mask, 1, keepdims=True)-1
        masked_reward = tf.math.multiply(disc_prob, gen_mask)
        print(masked_reward)
        log_prob = tf.math.log(tf.clip_by_value(gen_prob, 1e-20, 1.))
        masked_log_prob = tf.math.multiply(log_prob, gen_mask)
        G_loss = -tf.math.reduce_sum(tf.math.reduce_sum(tf.math.multiply(masked_log_prob, masked_reward), 1, keepdims=True))
        print(gen_prob, masked_log_prob)
        print(G_loss)
#        masked_reward = tf.reduce_sum(tf.math.multiply(disc_prob, gen_mask), 1, keepdims=True)
#
#        log_prob = tf.math.log(tf.clip_by_value(gen_prob, 1e-20, 1.))
#        masked_log_prob = tf.reduce_sum(tf.math.multiply(log_prob, gen_mask), 1, keepdims=True)
#        G_loss = -tf.math.reduce_mean(tf.math.multiply(masked_log_prob, masked_reward))
        
        return G_loss

    def G_train(self, spTgt=None):
        with tf.GradientTape() as tape:
            G_loss = self.rollout(spTgt=spTgt)
        grad = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_optim.apply_gradients(zip(grad, self.G.trainable_variables))
        return G_loss





















