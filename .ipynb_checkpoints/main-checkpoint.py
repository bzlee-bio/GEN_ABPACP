import tensorflow as tf
import numpy as np
import util
import Generator
import Trainer
import argparse
import shutil
import Discriminator

parser = argparse.ArgumentParser('Cond-PepSeqGAN Trainer v0.5')
parser.add_argument('--batch_size', type=int, help='Trainer batch size', default=64)
parser.add_argument('--G_units', help='List of generator unit sizes, example: 50,50 (two-layer stacked generator)', type=str)
parser.add_argument('--emb_dim', type=int, help='Dimesion of embedding layer', default=12)
parser.add_argument('--G_lr', type=float, help='G pre learning rate', default=None)

parser.add_argument('--G_iter', type=int, help='Number of G iter', default=None)



# parser.add_argument('--spTgt', type=str, help='SeqGAN specific targets', default=None)


args = parser.parse_args()
g_units = [int(x) for x in args.G_units.split(',')]

dataList_tr = {"AMP":"./Data/AMP_c_0.6_train.fasta",
                "ACP":"./Data/ACP_c_0.6_train.fasta"}
dataList_test = {"AMP":"./Data/AMP_c_0.6_test.fasta",
                "ACP":"./Data/ACP_c_0.6_test.fasta"}

params = {'maxSeqLen':30,
          'batch_size': args.batch_size,
          'G_units': g_units,
          'embed_dim':args.emb_dim,
          'G_pre_lr':args.G_lr,
          'G_p_iter': args.G_iter,
          'spTgt':args.spTgt,
          }
Data = util.DataHandler(dataList_tr, dataList_test, params)
G = Generator.Generator(params, Data)
mt = Trainer.modelTrainer(params, Data, G, None)
dirName = 'ALL'

for k, v in params.items():
    if v is not None:
        v_ = str(v).replace('[','').replace(']','').replace(' ','')
        dirName = f'{dirName}_{k}_{v_}'
print(dirName)
logger = util.logHandler(dirName)


seq, _, _, _ = G.genSeqs()

### Training Generator
for epoch in range(-params['G_p_iter'],0):
    logger.logStack(epoch,'G_Train/NLL_Summary', mt.G_pre_train().numpy())
    G.save_model_weight(logger.weightPath, epoch)
    for tgt, val in mt.G_NLL_test_loss().items():
        logger.logStack(epoch, f'G_Test/NLL_{tgt}', val.numpy())
    logger.write()




















