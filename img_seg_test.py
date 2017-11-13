import cluster_cls as clst
import numpy as np
import pylab as pl
from PIL import Image
import pa2 as fea_ex
import pickle
import scipy.io as scio


pic_l=['12003','21077','56028','62096','101087','117054','187029','208001','227046','299086'
       ,'310007','361010','361084','370036']
K_l=[5,8,8,4,4,7,7,8,4,4,6,5,8,8]




def EM_seg(pic,K):
    img = Image.open('PA2-cluster-images/images/'+pic+'.jpg')
    pl.subplot(1, 3, 1)
    pl.axis('off')
    pl.imshow(img)
    X, L = fea_ex.getfeatures(img, 7)
    lambda_km = 0.01
    l_type = ['b.','g.', 'r.', 'c.', 'm.','y.','k.','w.']
    cluster = clst.imgseg_cluster(X, K, l_type, 0, lambda_km)
    # Y_km,C_km=cluster.K_means()
    Y_EM, z_EM, C_EM = cluster.EM_GMM()
    # Y_ms,C_ms=cluster.mean_shift(h)
    # pickle.dump(Y_ms,open('Y_ms_'+str(hp)+'_'+str(hc)+'.txt','wb'))
    # pickle.dump(C_ms,open('C_ms_'+str(hp)+'_'+str(hc)+'.txt','wb'))
    segm = fea_ex.labels2seg(Y_EM, L)
    pl.subplot(1, 3, 2)
    pl.axis('off')
    pl.title('EM-GMM result')
    pl.imshow(segm)
    csegm = fea_ex.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.title('Number of clusters is '+str(K))
    pl.axis('off')
    pl.imshow(csegm)
    pl.savefig('EM_result/'+str(K)+'_'+pic+'.eps',dpi=400)



def K_means_seg(pic,K):
    img = Image.open('PA2-cluster-images/images/'+pic+'.jpg')
    pl.subplot(1, 3, 1)
    pl.axis('off')
    pl.imshow(img)
    X, L = fea_ex.getfeatures(img, 7)
    lambda_km = 0.01
    l_type = ['b.','g.', 'r.', 'c.', 'm.','y.','k.','w.']
    cluster = clst.imgseg_cluster(X, K, l_type, 0, lambda_km)
    Y_km,C_km=cluster.K_means()
    #Y_EM, z_EM, C_EM = cluster.K_means()
    # Y_ms,C_ms=cluster.mean_shift(h)
    # pickle.dump(Y_ms,open('Y_ms_'+str(hp)+'_'+str(hc)+'.txt','wb'))
    # pickle.dump(C_ms,open('C_ms_'+str(hp)+'_'+str(hc)+'.txt','wb'))
    segm = fea_ex.labels2seg(Y_km, L)
    pl.subplot(1, 3, 2)
    pl.axis('off')
    pl.title('K-means result')
    pl.imshow(segm)
    csegm = fea_ex.colorsegms(segm, img)
    pl.subplot(1, 3, 3)
    pl.title('Number of clusters is '+str(K))
    pl.axis('off')
    pl.imshow(csegm)
    pl.savefig('K_means/0dot01/'+str(K)+'_'+pic+'.eps',dpi=400)
#EM_seg('21077',6)
for i in range(len(pic_l)):
    K_means_seg(pic_l[i],K_l[i])

