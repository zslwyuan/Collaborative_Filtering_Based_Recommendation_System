import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy.random import RandomState
from sklearn.utils.multiclass import unique_labels
import time
from sklearn.model_selection import GridSearchCV
import argparse
import matplotlib.pyplot as plt
import sys
import pandas as pd

class MatrixEstimator(BaseEstimator):
    """ 
        Estimator for matrix factorization
    """
    def __init__(self, n_user=0, n_movie=0, K=2, batch_size=1500, learning_rate=0.003,
                 seed=1234, lambda_u=0.1,lambda_v=0.1, converge=1e-5,
                 max_rating=5, min_rating=1, do_shuffle=True,n_epochs=250,testset=np.array([])):
                 
        self.n_user = n_user
        self.n_movie = n_movie
        assert(n_user!=0)
        assert(n_movie!=0)
        self.K = K
        self.seed = seed
        self.random_state = RandomState(seed)
        self.do_shuffle = do_shuffle

        # batch size
        self.batch_size = batch_size

        # learning rate
        self.learning_rate = float(learning_rate)
        # regularization parameter
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.converge = converge
        self.max_rating = float(max_rating) \
            if max_rating is not None else max_rating
        self.min_rating = float(min_rating) \
            if min_rating is not None else min_rating

        # user/item features
        self.n_epochs=n_epochs
        self.testset=testset
        

    def fit(self, X,y):
        
       # print(self.get_params())
        #=== check the input and make sure the type of input X is integer ======
        X, y = check_X_y(X, y)
        if (X.shape[1]<2):
            print("The shape of input X is not correct! (This warning might occur\nduring check_estimator, because it generate input with random shape)")
            print("The shape of illegal input X is ",X.shape)
            return self

        ids = X
        ratings = y
        movie_ids = (ids[:,0]).astype(np.int64).T
        user_ids = (ids[:,1]).astype(np.int64).T 
        self.TIME_STAMP = []
        self.rmse_record = []
        self.test_rmse_record = []

        #===== initialization of feature matrices with random values ===========
        self.complete_user_features_ = self.random_state.rand(self.n_user, self.K)
        self.complete_movie_features_ = self.random_state.rand(self.n_movie, self.K)

        #==================== shuffle the data before fit ======================
        N_samples = user_ids.shape[0]
        if (self.do_shuffle):
            shuffle_ids = np.arange(N_samples)
            np.random.shuffle(shuffle_ids)
            movie_ids = movie_ids[shuffle_ids]
            user_ids = user_ids[shuffle_ids]
            ratings = ratings[shuffle_ids]
            ids = ids[shuffle_ids]

        #==================== calculation of basic parameters ==================
        self.mean_rating_samples_ = np.mean(ratings)
        last_rmse = None
        train_rmse = None
        batch_num = int(np.ceil(float(N_samples / self.batch_size)))

        #==================== initialization of gradients ======================
        u_feature_grads = np.zeros((self.n_user, self.K))
        v_feature_grads = np.zeros((self.n_movie, self.K))

        time_start=time.time()

        #=================== gradient descent algorithm ========================
        for epoch in range(self.n_epochs):

            for batch in range(batch_num):
               # print("dealing with batch(",batch,")")
                #============================== extract a batch from dataset ======================
                start_idx = int(batch * self.batch_size)
                end_idx = int((batch + 1) * self.batch_size)
                ratings_batch = ratings[start_idx:end_idx]
                user_ids_batch = user_ids[start_idx:end_idx]
                movie_ids_batch = movie_ids[start_idx:end_idx]
                size_of_the_batch = movie_ids_batch.shape[0]

                #======= extract corresponding rows in user/movie features for computation ========
                u_features = self.complete_user_features_.take(user_ids_batch, axis=0)   
                v_features = self.complete_movie_features_.take(movie_ids_batch, axis=0)

                # print(np.array_str(u_features, 100))
                # print(np.array_str(v_features, 100))
                #========================= computation of U^T*V ===================================
                preds = np.sum(u_features*v_features,axis=1)
                # print("prediction:")
                # print(np.array_str(preds, 200))
                # print(preds)
                # print("actual:")
                # print(ratings_batch - self.mean_rating_samples_)

                #====================== computation of U^T*V-Rij ==================================
                errs = preds - (ratings_batch - self.mean_rating_samples_)
                # print("error:")
                # print(errs)                
                #print("RMSE of current batch is ",sqrt(mean_squared_error(preds, (ratings_batch - self.mean_rating_samples_))))
                #============= extension error sequence into a matrix =============================
                err_mat = np.tile(errs, (self.K, 1)).T

                #============= extension error sequence into a matrix =============================
                u_grads = v_features * err_mat
                v_grads = u_features * err_mat
                u_feature_grads.fill(0.0)
                v_feature_grads.fill(0.0)

                #===== map the the first term of gradient to original gradient matrix =============
                for i in range(size_of_the_batch):
                    u_feature_grads[user_ids_batch[i], :] = u_feature_grads[user_ids_batch[i], :] + u_grads[i,:]
                    v_feature_grads[movie_ids_batch[i], :] = v_feature_grads[movie_ids_batch[i], :] + v_grads[i,:]

                #===== add the the second term of gradient to original gradient matrix ============
                u_feature_grads = u_feature_grads + self.lambda_u * self.complete_user_features_
                v_feature_grads = v_feature_grads + self.lambda_v * self.complete_movie_features_
          #      time.sleep( 1 )
                #================== update latent variables U and V================================
                self.complete_user_features_  -= self.learning_rate * u_feature_grads
                self.complete_movie_features_ -= self.learning_rate * v_feature_grads

             #   sys.stdout.write(' ' * 100 + '\r')
                sys.stdout.flush()
                if (train_rmse):
                    sys.stdout.write('\r'+"training @ batch%4d"%(batch)+" of epoch%4d"%(epoch) + ", current training RMSE is "+ str(train_rmse))
                else:
                    sys.stdout.write('\r'+"training @ batch%4d"%(batch)+" of epoch%4d"%(epoch) )
                sys.stdout.flush()

            # compute RMSE
            train_preds = self.predict(ids)
            train_rmse = sqrt(mean_squared_error(train_preds, ratings))
            self.TIME_STAMP.append(time.time()-time_start)
            self.rmse_record.append(train_rmse)
            if (self.testset.shape[0]!=0):
                test_preds = self.predict(self.testset[:,:2])
                test_rmse = sqrt(mean_squared_error(test_preds, self.testset[:,2]))
                self.test_rmse_record.append(test_rmse)
         #   print("========================",train_rmse,"====================================================================")
            # stop when converge
            if (last_rmse and abs(train_rmse - last_rmse) < self.converge):
                break
            else:
                last_rmse = train_rmse

        return self

    def predict(self, X):
        check_is_fitted(self, ['complete_user_features_', 'complete_movie_features_'])
        X = check_array(X)

        ids = X
        movie_ids = (ids[:,0]).astype(np.int64).T
        user_ids = (ids[:,1]).astype(np.int64).T 

        u_features = self.complete_user_features_.take(user_ids, axis=0)
        v_features = self.complete_movie_features_.take(movie_ids, axis=0)
        preds = np.sum(u_features * v_features, 1) + self.mean_rating_samples_
        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds










np.random.seed(1234)

#==================== argument controlling the output of figures ==================
parser = argparse.ArgumentParser(description='COMP5212 Programming Project 3')
parser.add_argument('--nofigure',  action="store_true",default=False,
                        help='Do not present figures during training')
parser.add_argument('--docv',  action="store_true",default=False,
                        help='Do cross-validation')
parser.add_argument('--sparse',  action="store_true",default=False,
                        help='Input is spare input')
parser.add_argument('--largeinput',  action="store_true",default=False,
                        help='Input is spare input')
args = parser.parse_args()
    
Show_figures = not args.nofigure
Do_cv = args.docv
Sparse_input = args.sparse
Large_input = args.largeinput

#==================== Load training and eval data ===============================
print("---Loading training and eval data")

if (Large_input):
    data_frame = pd.read_csv("ratings.dat", sep="::", usecols = [0, 1, 2], names = ['userID', 'movieID', 'rating'], engine = 'python')
    user_ids = data_frame.values[:,0]
    movie_ids = data_frame.values[:,1]
    ratings = data_frame.values[:,2].T
    N_samples = ratings.shape[0]
else:
    file = np.load("data.npz")
    N_samples = file['item_id'].shape[0]
    movie_ids = file['item_id'].reshape(N_samples)
    user_ids = file['user_id'].reshape(N_samples)
    ratings = file['rating'].reshape(N_samples).T

print("The number of input samples is ",N_samples)
print("---Loaded training and eval data\n")

#================= shuffle the data before spliting and fitting =================
shuffle_ids = np.arange(N_samples)
np.random.shuffle(shuffle_ids)
movie_ids = movie_ids[shuffle_ids]
user_ids = user_ids[shuffle_ids]
ratings = ratings[shuffle_ids]

#==================== get the number of users and movies ========================
n_user=max(user_ids)+1
n_movie=max(movie_ids)+1

#============ combine movie/user id as input of trainining ======================
inputX = np.array([movie_ids,user_ids]).T

#============ split dataset into training set and test set ======================
print("-- spliting dataset for training and test")

if (Sparse_input):
    print("spliting dataset for sparse-input training")
    train_partition = int(N_samples*0.2)
else:
    print("spliting dataset for dense-input training")
    train_partition = int(N_samples*0.8)

train_inputX = inputX[:train_partition,:]
test_inputX = inputX[train_partition:N_samples,:]
train_ratings = ratings[:train_partition]
test_ratings = ratings[train_partition:N_samples]
testset = np.array([test_inputX[:,0],test_inputX[:,1],test_ratings]).T
print("-- splited dataset for training and test, done\n")

if (Do_cv):

    n_jobs = int(input("During CV, multi-processing can be involved. Please specify the number of processes:"))
    #================== Initialization of estimator of PMF ==========================
    model0 = MatrixEstimator(n_user=n_user, n_movie=n_movie, K=2)

    #=================== CV for Lambda_u and Lambda_v ============================
    time_start=time.time()
    print("-- doing 5-fold CV to search lambda_u and lambda_v")
    parameters0 = { 'lambda_v':[0.1,1,10,100], \
                      'lambda_u':[0.1,1,10,100]}
    print("WARNING: during CV, the runtime output of training information might be not stable because multiple(%d"%n_jobs+") jobs are running at the same time.")
    clf0 = GridSearchCV(model0, parameters0,scoring='neg_mean_squared_error',cv=5,n_jobs=n_jobs)
    clf0.fit(train_inputX, train_ratings)
    print("\ncv_results_")
    print(clf0.cv_results_['mean_test_score'])
    print("\nBest paramters:")
    print(clf0.best_estimator_.get_params())
    time_end=time.time()
    print("the average time for CV process (one training+test process), based on multiple(%d"%n_jobs+") jobs, is "+str((time_end-time_start)/16/5)+"s")
    print("-- 5-fold CV to search lambda_u and lambda_v done\n")


    #======================== CV for K =================================
    time_start=time.time()
    print("-- doing 5-fold CV to search K")
    print("WARNING: during CV, the runtime output of training information might be not stable because multiple(%d"%n_jobs+") jobs are running at the same time.")
    parameters1 = {'K':[1,2,3,4,5]}
    model1 = MatrixEstimator(n_user=n_user, n_movie=n_movie,lambda_v=clf0.best_params_['lambda_v'],lambda_u=clf0.best_params_['lambda_u'])
    clf1 = GridSearchCV(model1, parameters1,scoring='neg_mean_squared_error',cv=5,n_jobs=n_jobs)
    clf1.fit(train_inputX, train_ratings)
    print("\ncv_results_")
    print(clf1.cv_results_['mean_test_score'])
    print("\nBest paramters:")
    print(clf1.best_estimator_.get_params())
    time_end=time.time()
    print("the average time for CV process (one training+test process), based on multiple(%d"%n_jobs+") jobs, is "+str((time_end-time_start)/5/5)+"s")
    print("-- 5-fold CV to search K\n")


    #============================= training ========================================
    print("-- training based on training dataset")
    time_start=time.time()
    model2 = MatrixEstimator(n_user=n_user, n_movie=n_movie,lambda_v=clf0.best_params_['lambda_v'],lambda_u=clf0.best_params_['lambda_u'],K=clf1.best_params_['K'],testset=testset)
    model2.fit(train_inputX, train_ratings)
    train_preds = model2.predict(train_inputX)
    train_rmse = sqrt(mean_squared_error(train_preds, train_ratings))
    print("\ntrain_rmse:")
    print(train_rmse)
    time_end=time.time()
    print("the time for training is "+str((time_end-time_start))+"s")
    print("-- training done\n")

    #================================ test =========================================
    print("-- testing")
    time_start=time.time()
    test_preds = model2.predict(test_inputX)
    test_rmse = sqrt(mean_squared_error(test_preds, test_ratings))
    print("test_rmse:")
    print(test_rmse)
    time_end=time.time()
    print("the time for test is "+str((time_end-time_start))+"s")
    print("-- tested\n")

    #======================== ploting performance =================================
    if (Show_figures):
        colors = ['r','y']
        legends = {'Training':'o','Test':'*'}
        plt.plot(model2.TIME_STAMP,model2.rmse_record,'r-')
        if (testset.shape[0]>0):
            plt.plot(model2.TIME_STAMP,model2.test_rmse_record,'b-')
        plt.legend(legends.keys(),loc=1)
        plt.grid()
        plt.xlabel('time(s)')
        plt.ylabel('RMSE')
        plt.show()

else:
    
    #============================= training ========================================
    print("-- training based on training dataset")
    time_start=time.time()
    model0 = MatrixEstimator(n_user=n_user, n_movie=n_movie,lambda_u=0.1,lambda_v=0.1,K=3,learning_rate=0.003,testset=testset)
    if (Large_input):
        model0 = MatrixEstimator(n_user=n_user, n_movie=n_movie,lambda_u=0.1,lambda_v=0.1,K=100,learning_rate=0.001,batch_size=15000,testset=testset)
    if (Sparse_input):
        model0 = MatrixEstimator(n_user=n_user, n_movie=n_movie,lambda_u=0.1,lambda_v=1,K=5,learning_rate=0.003,testset=testset)
    model0.fit(train_inputX, train_ratings)
    train_preds = model0.predict(train_inputX)
    train_rmse = sqrt(mean_squared_error(train_preds, train_ratings))
    print("\ntrain_rmse:")
    print(train_rmse)
    time_end=time.time()
    print("the time for training is "+str((time_end-time_start))+"s")
    print("-- training done\n")

    #================================ test =========================================
    print("-- testing")
    time_start=time.time()
    test_preds = model0.predict(test_inputX)
    test_rmse = sqrt(mean_squared_error(test_preds, test_ratings))
    print("test_rmse:")
    print(test_rmse)
    time_end=time.time()
    print("the time for test is "+str((time_end-time_start))+"s")
    print("-- tested\n")

    #======================== ploting performance =================================
    if (Show_figures):
        colors = ['r','y']
        legends = {'Training':'o','Test':'*'}
        plt.plot(model0.TIME_STAMP,model0.rmse_record,'r-')
        if (testset.shape[0]>0):
            plt.plot(model0.TIME_STAMP,model0.test_rmse_record,'b-')
        plt.legend(legends.keys(),loc=1)
        plt.grid()
        plt.xlabel('time(s)')
        plt.ylabel('RMSE')
        plt.show()
