import cluster_cls as clst
import numpy as np
import pylab as pl
from PIL import Image
import pa2 as fea_ex
import pickle

img = Image.open('PA2-cluster-images/images/187029.jpg')
X,L=fea_ex.getfeatures(img,7)


K=4
lambda_km=0.5
hp=20.0
hc=10.0
h=np.array([[hp],[hc]])
cluster=clst.imgseg_cluster(X,K,0,0,lambda_km,hp,hc)
#Y_km,C_km=cluster.K_means()
#Y_EM,z_EM,C_EM=cluster.EM_GMM()
Y_ms,C_ms=cluster.mean_shift(h)
pickle.dump(Y_ms,open('Y_ms.txt','wb'))
pickle.dump(C_ms,open('C_ms.txt','wb'))
segm = fea_ex.labels2seg(Y_ms, L)
pl.subplot(1,2,1)
pl.imshow(segm)
csegm = fea_ex.colorsegms(segm, img)
pl.subplot(1, 2, 2)
pl.imshow(csegm)
pl.savefig('mean_shift.eps')
pl.show()

