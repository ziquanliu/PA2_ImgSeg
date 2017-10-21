import numpy as np
import matplotlib.pyplot as plt
import copy
import ctypes
from ctypes import cdll
import pickle
mydll=cdll.LoadLibrary('cal_kernel.so')
cal_k=mydll.cal_kernel
cal_k.restype=ctypes.c_double


def k_mean_dist(x,mean_value,lambda_km):
    left=np.sum(np.square(x[0:2]-mean_value[0:2]))
    right=np.sum(np.square(x[2:4]-mean_value[2:4]))
    return left+lambda_km*right



def squ_Euc_dist(x, mean_value):
    return np.sum(np.square(x - mean_value))


def cal_gaussian(x,miu,Cov):
    exp_f=-(x-miu).transpose().dot(np.linalg.inv(Cov)).dot(x-miu)/2.0
    #print 'x shape',(x-miu).shape
    return np.exp(exp_f)/np.sqrt(np.linalg.det(Cov))

def cal_log_gaussian(x,miu,Cov):
    L=np.linalg.cholesky(Cov)
    left=-np.sum(np.log(np.diagonal(L)))
    right=-(x-miu).transpose().dot(np.linalg.inv(Cov)).dot(x-miu)/2.0
    return left+right

def cal_kernel(x,x_old,h_c,h_p):
    left = -np.sum(np.square(x[0:2] - x_old[0:2]))*0.5/float(h_c**2)
    right = -np.sum(np.square(x[2:4] - x_old[2:4]))*0.5/float(h_p**2)
    return np.exp(left+right)





def gene_cov(dim):
    cov=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i,dim):
            if i==j:
                cov[i,j]=5.0+np.random.rand()
            else:
                cov[i,j]=0
                cov[j,i]=cov[i,j]
    return cov


def cal_new_miu(j,N,z,X):
    sam_num=X.shape[1]
    s=np.zeros(X[:,0].shape)
    for i in range(sam_num):
        s+=z[i,j]*X[:,i]
    return s/N


def cal_new_cov(j,N,z,X,miu_j):
    sam_num=X.shape[1]
    sam_dim=X.shape[0]
    Sig_s=np.zeros((sam_dim,sam_dim))
    for i in range(sam_num):
        temp=(X[:,i]-miu_j).reshape(sam_dim,1)
        Sig_s+=z[i,j]*temp.dot(temp.transpose())
    return Sig_s/N

def cluster_plt(line_type,z,X,K,name,x_mean,h=0):
    dim = X.shape[0]
    num = X.shape[1]
    f1 = plt.figure()
    z_t=z.copy()
    for i in range(num):
        max_index=np.argmax(z_t[i,:])
        z_t[i,:]=np.zeros((1,K))
        z_t[i,max_index]=1

    for j in range(K):
        num_k = int(np.sum(z_t[:, j]))
        X_temp = np.zeros((dim, num_k))
        ind_temp = 0
        for i in range(num):
            if z_t[i, j] == 1:
                X_temp[:, ind_temp] = X[:, i]
                ind_temp += 1
        plt.plot(X_temp[0, :], X_temp[1, :], line_type[j])
    plt.plot(x_mean[0, :], x_mean[1, :], 'kx')
    plt.title(name)
    plt.savefig(name+'/image/cluster_result_h'+str(h)+'.eps',dpi=300)
    plt.show()

def plt_save(line_type,z,X,K,name,x_mean=None,h=0):
    cluster_plt(line_type,z,X,K,name,x_mean,h)
    pickle.dump(z,open(name+'/data/'+'cluster_result_h'+str(h)+'.txt','wb'))
    pickle.dump(z, open(name+'/data/' +'cluster_mean_h' + str(h) + '.txt', 'wb'))

def update_mean(x_old,DS,Sigma):
    nominator=np.zeros(x_old.shape)
    denom=0.0
    num_DS=DS.shape[1]
    dim=DS.shape[0]
    for i in range(num_DS):
        g_kern=cal_gaussian(DS[:,i].reshape(dim,1),x_old,Sigma)
        nominator+=(DS[:,i].reshape(dim,1))*g_kern
        denom+=g_kern
    return nominator/denom

def update_mean_img(x_old,DS,h_c,h_p):
    nominator=np.zeros(x_old.shape)
    denom=0.0
    num_DS=DS.shape[1]
    dim=DS.shape[0]
    for i in range(num_DS):
        g_kern=cal_kernel(DS[:,i].reshape(dim,1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          x_old.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_double(h_c),
                          ctypes.c_double(h_p))
        nominator+=(DS[:,i].reshape(dim,1))*g_kern
        denom+=g_kern
    return nominator/denom


def ms_find_cluster(h,X_Mean,X):
    dim = X.shape[0]
    num = X.shape[1]
    cluster_cen = [X_Mean[:, 0].reshape((dim, 1))]
    if isinstance(h,np.ndarray):
        MIN_MEAN_DEV = np.sum(h)/2.0
    else:
        MIN_MEAN_DEV=h
    num_cen = 1
    print num
    for i in range(num):
        temp = np.zeros((num_cen, 1))
        for j in range(num_cen):
            temp[j, 0] = np.sum(np.absolute(X_Mean[:, i].reshape((dim, 1)) - cluster_cen[j]))
        if np.min(temp) > MIN_MEAN_DEV:
            cluster_cen.append(X_Mean[:, i].reshape((dim, 1)))
            num_cen += 1
        print 'number of centers',num_cen
        if num_cen>10: break

    z = np.zeros((num, num_cen))
    for i in range(num):
        temp = np.zeros((num_cen, 1))
        for j in range(num_cen):
            temp[j, 0] = np.sum(np.absolute(X_Mean[:, i].reshape((dim, 1)) - cluster_cen[j]))
        z[i, np.argmin(temp)] = 1
    miu=np.zeros((dim,num_cen))
    for i in range(num_cen):
        miu[:,i]=cluster_cen[i].reshape((dim,))
    return z,num_cen,miu

def ms_find_cluster_n(h,X_Mean,X):
    dim = X.shape[0]
    num = X.shape[1]
    if isinstance(h,np.ndarray):
        MIN_MEAN_DEV = np.sum(h)/2.0
    else:
        MIN_MEAN_DEV=h
    num_mean=num
    num_cen = 0
    cls_index=[]
    mean=X_Mean.copy()
    miu_l=[]
    for i in range(num_mean):
        if mean[int(dim * np.random.rand()),i] == 0 and mean[int(dim * np.random.rand()),i] == 0:
            continue
        mean_l_ind=[i]
        miu=mean[:,i].reshape((dim,1))
        for j in range(num_mean):
            if i==j: continue
            if mean[int(dim*np.random.rand()),j]==0 and mean[int(dim*np.random.rand()),j]==0:
                continue
            if np.sum(np.absolute(mean[:, j].reshape((dim, 1)) - mean[:, i].reshape((dim, 1))))<MIN_MEAN_DEV:
                mean_l_ind.append(j)
                mean[:,j]=np.zeros((dim))
                miu+=mean[:, j].reshape((dim, 1))
        miu_l.append(miu/float(len(mean_l_ind)))
        cls_index.append(mean_l_ind)
        num_cen+=1
    z = np.zeros((num, num_cen))
    print len(cls_index)
    print num_cen
    for i in range(num_cen):
        for j in range(len(cls_index[i])):
            z[cls_index[i][j],i]=1
    return z,num_cen,miu_l



class cluster(object):
    def __init__(self,X,K,l_type,true_label):
        self.X=X
        self.K=K
        self.line_type=l_type
        self.true_X=true_label



    def K_means(self):
        # initialize miu_j
        dim = self.X.shape[0]
        num = self.X.shape[1]
        miu = np.zeros((dim, self.K))
        for i in range(self.K):
            miu[:, i] = self.X[:, int(num * np.random.rand())]

        # initialize z_ij
        z = np.zeros((num, self.K))
        for i in range(num):
            min_distance = squ_Euc_dist(self.X[:, i], miu[:, 0])
            ind = 0
            for j in range(1, self.K):
                distance = squ_Euc_dist(self.X[:, i], miu[:, j])
                if distance < min_distance:
                    min_distance = distance
                    ind = j
            z[i, ind] = 1
        #cluster_plt(self.line_type, z, self.X,self.K)

        # iteration
        center_shift = 100
        min_shift = 10 ** -10
        iter_n = 0
        miu_new = np.zeros((dim, self.K))

        while center_shift > min_shift:
            print 'iteration number', iter_n
            iter_n += 1
            for j in range(self.K):
                denom = 0
                for i in range(num):
                    denom += float(z[i, j]) * self.X[:, i]
                miu_new[:, j] = denom / float(np.sum(z[:, j]))
            center_shift = np.sum(np.absolute(miu_new - miu))
            #print 'center shift:', center_shift
            miu = miu_new.copy()
            z = np.zeros((num, self.K))
            for i in range(num):
                min_distance = squ_Euc_dist(self.X[:, i], miu[:, 0])
                ind = 0
                for j in range(1, self.K):
                    distance = squ_Euc_dist(self.X[:, i], miu[:, j])
                    if distance < min_distance:
                        min_distance = distance
                        ind = j
                z[i, ind] = 1
            #cluster_plt(self.line_type, z, self.X, self.K)
        #cluster_plt(self.line_type, z, self.X, self.K,miu)
        return z,miu


    def EM_GMM(self):
        dim = self.X.shape[0]
        num = self.X.shape[1]
        miu = np.zeros((dim, self.K))
        Sigma = []
        for i in range(self.K):
            miu[:, i] = self.X[:, int(num * np.random.rand())]  # initialize miu
            Sigma.append(gene_cov(dim))  # initialize Sigma

        pi = np.ones((1, self.K)) / float(self.K)  # initialize pi

        z = np.zeros((num, self.K))  # initialize z
        resid = 10
        MIN_RES = 10 ** -10
        pi_new = pi.copy()
        miu_new = miu.copy()
        Sigma_new = copy.copy(Sigma)
        iter_num = 0

        while resid > MIN_RES:

            print 'iteration number', iter_num
            iter_num += 1
            for i in range(num):
                compnt = np.zeros((1, self.K))
                for j in range(self.K):
                    compnt[0, j] = cal_log_gaussian(self.X[:, i].reshape(dim,1), miu[:, j].reshape(dim,1), Sigma[j])

                l_star=np.max(compnt)
                compnt_new=compnt-l_star
                l=l_star+np.log(np.sum(np.exp(compnt_new)))
                for j in range(self.K):
                    gama=compnt[0,j]-l
                    z[i,j]=np.exp(gama)
            # print z
            #cluster_plt(self.line_type, z, self.X, self.K)
            # print np.sum(z,0)
            N = np.sum(z, 0).reshape((1, self.K))
            for j in range(self.K):
                pi_new[0, j] = N[0, j] / num
                miu_new[:, j] = cal_new_miu(j, N[0, j], z, self.X)
                Sigma_new[j] = cal_new_cov(j, N[0, j], z, self.X, miu_new[:, j])
                # break
            # break
            resid = np.sum(np.abs(pi - pi_new)) + np.sum(np.abs(miu - miu_new))
            print 'resid', resid
            pi = pi_new.copy()
            miu = miu_new.copy()
            Sigma = copy.copy(Sigma_new)
        #cluster_plt(self.line_type, z, self.X, self.K, miu)
        z_t = z.copy()
        for i in range(num):
            max_index = np.argmax(z_t[i, :])
            z_t[i, :] = np.zeros((1, self.K))
            z_t[i, max_index] = 1
        Y=np.zeros((num,1))
        for i in range(num):
            Y[i,0]=np.argmax(z[i,:])+1
        print 'This is cluster EM-GMM'
        return Y,z_t,miu

    def mean_shift(self,h):
        dim = self.X.shape[0]
        num = self.X.shape[1]
        Sigma = h ** 2 * np.eye(dim, dim)
        shift_ind = np.ones((1, num))
        MIN_RES = 10 ** -2
        x_mean = self.X.copy()
        iter_num = 0
        while np.sum(shift_ind) > 0:
            print 'iteration number', iter_num
            iter_num += 1
            for i in range(num):
                if shift_ind[:, i] == 0: continue
                x_old = x_mean[:, i].reshape(dim, 1)
                x_new = update_mean(x_old, self.X, Sigma)
                if np.max(np.abs(x_old - x_new)) < MIN_RES:
                    shift_ind[:, i] = 0
                else:
                    x_mean[:, i] = x_new.reshape((dim))
            if iter_num > 10 ** 4:
                break

        z, K, clst_mean= ms_find_cluster(h, x_mean, self.X)

        print 'When h is ',h
        print 'Number of clusters is ', K
        if K==4:
            #cluster_plt(self.line_type,z,self.X,K,x_mean,h)
            #pickle.dump(z, open('data/h_' + str(h) + '_mean_shift_z.txt', 'wb'))
            #pickle.dump(x_mean, open('data/h_' + str(h) + '_mean_shift_mean.txt', 'wb'))
            a=~0
        else:
            print "Number of clusters is not equal to 4!"
        return z,clst_mean




class imgseg_cluster(cluster):
    def __init__(self,X,K,l_type,true_label,lambdakm,hp,hc):
        cluster.__init__(self,X,K,l_type,true_label)
        self.lambda_km=lambdakm


    def K_means(self):
        dim = self.X.shape[0]
        num = self.X.shape[1]
        miu = np.zeros((dim, self.K))
        for i in range(self.K):
            miu[:, i] = self.X[:, int(num * np.random.rand())]

        # initialize z_ij
        z = np.zeros((num, self.K))
        for i in range(num):
            min_distance = k_mean_dist(self.X[:, i], miu[:, 0],self.lambda_km)
            ind = 0
            for j in range(1, self.K):
                distance = k_mean_dist(self.X[:, i], miu[:, j],self.lambda_km)
                if distance < min_distance:
                    min_distance = distance
                    ind = j
            z[i, ind] = 1
        # cluster_plt(self.line_type, z, self.X,self.K)

        # iteration
        center_shift = 100
        min_shift = 10 ** -10
        iter_n = 0
        miu_new = np.zeros((dim, self.K))

        while center_shift > min_shift:
            print 'iteration number', iter_n
            iter_n += 1
            for j in range(self.K):
                denom = 0
                for i in range(num):
                    denom += float(z[i, j]) * self.X[:, i]
                miu_new[:, j] = denom / float(np.sum(z[:, j]))
            center_shift = np.sum(np.absolute(miu_new - miu))
            # print 'center shift:', center_shift
            miu = miu_new.copy()
            z = np.zeros((num, self.K))
            for i in range(num):
                min_distance = squ_Euc_dist(self.X[:, i], miu[:, 0])
                ind = 0
                for j in range(1, self.K):
                    distance = squ_Euc_dist(self.X[:, i], miu[:, j])
                    if distance < min_distance:
                        min_distance = distance
                        ind = j
                z[i, ind] = 1
        Y=np.zeros((num,1))
        for i in range(num):
            Y[i,0]=np.argmax(z[i,:])+1
        print 'This is imgseg k-means'
        return Y, miu



    def mean_shift(self,h):
        dim = self.X.shape[0]
        num = self.X.shape[1]
        print num
        h_c=h[0,0]
        h_p=h[1,0]
        shift_ind = np.ones((1, num))
        MIN_RES = 10 ** -2
        x_mean = self.X.copy()
        iter_num = 0
        while np.sum(shift_ind) > 0:
            print 'iteration number', iter_num
            iter_num += 1
            for i in range(num):
                if shift_ind[:, i] == 0: continue
                x_old = x_mean[:, i].reshape(dim, 1)
                x_new = update_mean_img(x_old, self.X, h_c,h_p)
                if np.max(np.abs(x_old - x_new)) < MIN_RES:
                    shift_ind[:, i] = 0
                else:
                    x_mean[:, i] = x_new.reshape((dim))
            if iter_num > 10 ** 4:
                break
            print 'unshift',shift_ind.sum()


        z, K, clst_mean= ms_find_cluster_n(h, x_mean, self.X)
        print 'Number of clusters is ', K
        Y=np.zeros((num,1))
        for i in range(num):
            Y[i,0]=np.argmax(z[i,:])+1
        print 'This is imgseg mean shift'
        return Y,clst_mean





