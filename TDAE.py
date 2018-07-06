import scipy.io as sio
import os
from scipy.sparse import csr_matrix
import numpy as np
from MatUtils import matrix_cross_validation, negdict_mat
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli
from RecEval import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.random.seed(2018)

class TDAE():
    def __init__(self,sess,num_factors=140,epoch=2000,topK=[5,10,15,20,25],T=1000,lr=0.01,q=0.2,alpha=0.8,lambdaT=0.01,lambdaC=0.01,
                 beta=0.01,batch_size=128):
        self.session=sess
        self.num_factors=num_factors
        self.skip_step=T
        self.lr=lr
        self.epoch=epoch
        self.topK=topK
        self.q=q
        self.alpha=alpha
        self.lambdaT=lambdaT
        self.lambdaC=lambdaC
        self.beta=beta
        self.batch_size=batch_size

    def prepare_data(self,o_matrix,train_data,test_data,trust_data):
        self.num_user, self.num_item=o_matrix.shape
        self.train_array=train_data.toarray()
        _, self.ranking_dict,self.test_dict=negdict_mat(o_matrix,test_data, mod='precision',
                                                        num_neg=0, random_state=None)
        self.num_training=self.train_array.shape[0]
        self.num_batch=int(self.num_training/self.batch_size)
        self.trust_data=trust_data.toarray()
        print('Data Completed')

    def build_model(self):
        with tf.variable_scope('TDAE_Model',reuse=tf.AUTO_REUSE):
            self.rating=tf.placeholder(dtype=tf.float32,shape=[None, self.num_item],name='rating')
            self.trust=tf.placeholder(dtype=tf.float32, shape=[None, self.num_user],name='trust')
            Theta0 =tf.get_variable(name='Theta0',shape=[self.num_factors],
                                    initializer=tf.truncated_normal_initializer(mean=0,stddev=0.03))
            Theta1=tf.get_variable(name='Theta1',shape=[self.num_factors],
                                    initializer=tf.truncated_normal_initializer(mean=0,stddev=0.03))
            # Corrupted
            R_berngen=Bernoulli(probs=self.q,dtype=self.rating.dtype)
            Rcorrupt_mask=R_berngen.sample(tf.shape(self.rating))
            Rcorrupt_input=tf.multiply(self.rating,Rcorrupt_mask)
            T_berngen=Bernoulli(probs=self.q,dtype=self.trust.dtype)
            Tcorrupt_mask=T_berngen.sample(tf.shape(self.trust))
            Tcorrupt_input=tf.multiply(self.trust,Tcorrupt_mask)
            # Encoder
            Rlayer1_w=tf.get_variable(name='Re_weights',shape=[self.num_item,self.num_factors],
                                      initializer=tf.truncated_normal_initializer(mean=0,stddev=0.03))
            Rlayer1_b=tf.get_variable(name='Re_bias',shape=[self.num_factors],initializer=tf.zeros_initializer())

            Tlayer1_w=tf.get_variable(name='Te_weights',shape=[self.num_user,self.num_factors],
                                      initializer=tf.truncated_normal_initializer(mean=0,stddev=0.03))
            Tlayer1_b=tf.get_variable(name='Te_bias',shape=[self.num_factors],initializer=tf.zeros_initializer())

            Rlayer1=tf.sigmoid(tf.matmul(Rcorrupt_input,Rlayer1_w)+Rlayer1_b)
            Tlayer1=tf.sigmoid(tf.matmul(Tcorrupt_input,Tlayer1_w)+Tlayer1_b)
            layerP=self.alpha*Rlayer1+(1-self.alpha)*Tlayer1
            # Decoder
            Rlayer2_w=tf.get_variable(name='Rd_weights',shape=[self.num_factors, self.num_item],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03))
            Rlayer2_b=tf.get_variable(name='Rd_bias',shape=[self.num_item],initializer=tf.zeros_initializer())

            Tlayer2_w=tf.get_variable(name='Td_weights',shape=[self.num_factors, self.num_user],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.03))
            Tlayer2_b=tf.get_variable(name='Td_bias',shape=[self.num_user],
                                    initializer=tf.zeros_initializer())
            self.Pred_R=tf.sigmoid(tf.matmul(layerP,Rlayer2_w)+Rlayer2_b)
            self.Pred_T=tf.sigmoid(tf.matmul(layerP,Tlayer2_w)+Tlayer2_b)
            #loss
            loss1=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating,logits=self.Pred_R))
            loss2=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.trust,logits=self.Pred_T))
            loss3=self.lambdaT*(tf.nn.l2_loss(Rlayer1_w)+tf.nn.l2_loss(Rlayer1_b)+tf.nn.l2_loss(Tlayer1_w)
                                +tf.nn.l2_loss(Tlayer1_b)+tf.nn.l2_loss(Tlayer2_b)+tf.nn.l2_loss(Tlayer2_w)
                                +tf.nn.l2_loss(Rlayer2_b)+tf.nn.l2_loss(Rlayer2_w))
            loss4=tf.nn.l2_loss(Theta0)+tf.nn.l2_loss(Theta1)
            loss5=2*(tf.nn.l2_loss(Rlayer1-tf.multiply(Tlayer1,Theta0))+
                     tf.nn.l2_loss(Tlayer1-tf.multiply(Rlayer1,Theta1)))
            self.loss=loss1+loss2+loss3+loss4+loss5
            self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        return self

    def train_one_epoch(self,epoch):
        random_idx=np.random.permutation(self.num_user)
        n_batches,total_loss=0,0
        for i in range(self.num_batch):
            if i==self.num_batch-1:
                batch_idx = random_idx[i * self.batch_size:]
                batch_ratings = self.train_array[batch_idx, :]
                batch_trust=self.trust_data[batch_idx,:]
            else:
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]
                batch_ratings = self.train_array[batch_idx, :]
                batch_trust=self.trust_data[batch_idx,:]

            _, l=self.session.run([self.opt,self.loss],
                                  feed_dict={self.rating:batch_ratings,self.trust:batch_trust})
            n_batches+=1
            total_loss+=1

            if n_batches % self.skip_step==0:
                print("Training Epoch {0} Batch {1}: [Loss] = {2}"
                          .format(epoch, n_batches, total_loss / n_batches))

    def eval_one_epoch(self,epoch):
        pred_y,pred_t=self.session.run([self.Pred_R, self.Pred_T],
                                feed_dict={self.rating: self.train_array, self.trust:self.trust_data})

        pred_y=pred_y.clip(min=0,max=1)
        pred_t=pred_t.clip(min=0,max=1)
        metric_matrix=np.zeros((4,5))
        for index,t in enumerate(self.topK):
            n_batches, total_pre, total_recall, total_ndcg= 0, 0, 0, 0
            for u in self.ranking_dict:
                iid = self.ranking_dict[u]
                rk = pred_y[u, :][np.array(iid)]
                pre,recall,ndcg=rankingMetrics(rk, iid, t, self.test_dict[u])
                n_batches += 1
                total_pre+=pre
                total_recall+=recall
                total_ndcg+=ndcg
            ave_pre, ave_recall, ave_ndcg=total_pre/n_batches, total_recall/n_batches,total_ndcg/n_batches
            ave_F=2*ave_pre*ave_recall/(ave_pre+ave_recall)
            total_metric=np.array([ave_pre,ave_recall,ave_F,ave_ndcg])
            metric_matrix[:,index]=total_metric
        print(metric_matrix)

    def train(self,restore=False,save=False,datafile=None):
        if restore:  # Restore the model from checkpoint
            self.restore_model(datafile, verbose=True)
        else:
            self.session.run(tf.global_variables_initializer())

        if not save:
            self.eval_one_epoch(-1)
            for i in range(self.epoch):
                self.train_one_epoch(i)
                self.eval_one_epoch(i)
        else:
            _, _, previous_ndcg = self.eval_one_epoch(-1)
            for i in range(self.epoch):
                self.train_one_epoch(i)
                _, _, ndcg = self.eval_one_epoch(i)
                if ndcg < previous_ndcg:
                    previous_ndcg = ndcg
                    self.save_model(datafile, verbose=False)

    def save_model(self, datafile, verbose=False):
        saver = tf.train.Saver()
        path = saver.save(self.session, datafile)
        if verbose:
            print("Model Saved in Path: {0}".format(path))

    def restore_model(self, datafile, verbose=False):
        saver = tf.train.Saver()
        saver.restore(self.session, datafile)
        if verbose:
            print("Model Restored from Path: {0}".format(datafile))

    def evaluate(self, datafile):
        self.restore_model(datafile, True)
        self.eval_one_epoch(-1)

if __name__=="__main__":
    Mat1=sio.loadmat('Fc_rating2')
    rating=Mat1['rating']
    Mat2=sio.loadmat('FCtrust2')
    trust=Mat2['FCtrust2']
    trust_value=np.ones((trust.shape[0],1))
    trust=np.hstack((trust,trust_value))
    # form rating matrix
    R_row = rating[:, 0]
    R_col = rating[:, 1]
    R_value = rating[:, 2]
    num_user = np.max(R_row) + 1
    num_item = np.max(R_col) + 1
    R_matrix = csr_matrix((R_value, (R_row, R_col)), shape=(num_user, num_item)).tocoo()
    # form trust matrix
    T_row=trust[:,0]
    T_col=trust[:,1]
    T_value=trust[:,2]
    T_matrix=csr_matrix((T_value,(T_row,T_col)),shape=(num_user,num_user)).tocoo()
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=8,
                                          inter_op_parallelism_threads=8,
                                          gpu_options=gpu_options)) as sess:
        train_data, test_data = matrix_cross_validation(R_matrix, 5, random_state=None)
        model=TDAE(sess)
        model.prepare_data(o_matrix=R_matrix,train_data=train_data,test_data=test_data,trust_data=T_matrix)
        model.build_model()
        model.train(restore=False,save=False,datafile='TDAE.ckpt')






