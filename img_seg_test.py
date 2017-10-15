import cluster_cls as clst
import pa2
import numpy as np
import pylab as pl
from PIL import Image
import pa2 as fea_ex


img = Image.open('PA2-cluster-images/images/12003.jpg')
X,L=fea_ex.getfeatures(img,7)
K=3
lambda_km=0.5
hp=2
hc=1.7
cluster=clst.imgseg_cluster(X,K,0,0,lambda_km,hp,hc)
Y_km,C_km=cluster.K_means()
Y_EM,z_EM,C_EM=cluster.EM_GMM()