import pa2 as clst
from PIL import Image
import numpy as np
import pickle



Y=pickle.load(open('Y_ms_6.txt','rb'))
C=pickle.load(open('C_ms_6.txt','rb'))
x_mean=pickle.load(open('x_mean.txt','rb'))
num=x_mean.shape[1]
for i in range(num):
    print x_mean[:,i]
