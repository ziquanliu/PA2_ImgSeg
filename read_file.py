import pa2 as clst
from PIL import Image
import numpy as np


img = Image.open('PA2-cluster-images/images/12003.jpg')

X,L=clst.getfeatures(img,7)
#print img.size
print L
print X.shape
A=np.array([0,1,2,3])

print isinstance(A,np.ndarray)