import cluster_cls as clst
import numpy as np
import pylab as pl
from PIL import Image
import pa2 as fea_ex
import pickle
import scipy.io as scio






img = Image.open('PA2-cluster-images/images/21077.jpg')
pl.subplot(1, 3, 1)
pl.imshow(img)
X,L=fea_ex.getfeatures(img,7)


K=10
lambda_km=0.01
hp=24.0

hc=12.0
h=np.array([[hp],[hc]])
l_type=['r.','b.','g.','y.']
cluster=clst.imgseg_cluster(X,K,l_type,0,lambda_km,hp,hc)
#Y_km,C_km=cluster.K_means()
Y_EM,z_EM,C_EM=cluster.EM_GMM()
#Y_ms,C_ms=cluster.mean_shift(h)
#pickle.dump(Y_ms,open('Y_ms_'+str(hp)+'_'+str(hc)+'.txt','wb'))
#pickle.dump(C_ms,open('C_ms_'+str(hp)+'_'+str(hc)+'.txt','wb'))
segm = fea_ex.labels2seg(Y_EM, L)
pl.subplot(1,3,2)
pl.imshow(segm)
csegm = fea_ex.colorsegms(segm, img)
pl.subplot(1, 3, 3)
pl.imshow(csegm)
#pl.savefig('ms_'+str(K)+'.eps')
pl.show()


