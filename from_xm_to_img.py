import cluster_cls as clst
import numpy as np
import pylab as pl
from PIL import Image
import pa2 as fea_ex
import pickle
import scipy.io as scio
pic='117054'

img = Image.open('PA2-cluster-images/images/'+pic+'.jpg')
pl.subplot(1, 3, 1)
pl.axis('off')
pl.title('original image',fontsize=10)
pl.imshow(img)
X,L=fea_ex.getfeatures(img,7)
num = X.shape[1]
hp=16.0
hc=10.0
h=np.array([[hp],[hc]])
x_mean=pickle.load(open(pic+'/x_mean_'+str(hp)+'_'+str(hc)+'_'+pic+'.txt','rb'))


z, K, clst_mean = clst.ms_find_cluster_n(h, x_mean, X)
# cluster_plt(self.line_type,z,self.X[0:2,:].reshape((2,num)),self.K,'ms',x_mean[0:2,:])

print 'Number of clusters is ', K
Y = np.zeros((num, 1))
for i in range(num):
    Y[i, 0] = np.argmax(z[i, :]) + 1
print 'This is imgseg mean shift'
segm = fea_ex.labels2seg(Y, L)
pl.subplot(1,3,2)
pl.axis('off')
pl.title('Number of clusters is '+str(K),fontsize=10)
pl.imshow(segm)
csegm = fea_ex.colorsegms(segm, img)
pl.subplot(1, 3, 3)
pl.axis('off')
pl.imshow(csegm)
pl.title('hp='+str(hp)+',hc='+str(hc),fontsize=10)
pl.savefig(pic+'/ms_hp='+str(hp)+'_hc='+str(hc)+'.eps',dpi=400)
pl.show()