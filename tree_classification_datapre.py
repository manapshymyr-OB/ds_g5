#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the libraries you need

import os
import math
import numpy as np
import rasterio
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
from matplotlib.image import imread
from itertools import product
from PIL import Image
from itertools import chain
import json
from jsonpath import jsonpath 

from matplotlib.colors import Normalize

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target

import tensorflow as tf
import torch
# Folium setup.
import folium


# In[8]:


path = '/Users/siruiwang/Documents/ESPACE-LECTURE/3rd_semester/Data science of earth observation/Project_data_science/data0113/GeoJson/untitled folder/Betula_pendula_0113.geojson'
index=5
tree_type = 'Betula_pendula'
kernel=9
bands=40
with open(path) as f:
    data = json.load(f)
for feature in data['features']:
    for properties in feature['properties']:
        print(properties)

B8 = jsonpath(data, "$..B8") 

B8data=np.array(B8[0])
print(B8data.shape)
plt.imshow(B8data) 
#print(ddate) 


# In[9]:


B2 = jsonpath(data, "$..B2") 
B3 = jsonpath(data, "$..B3") 
B4 = jsonpath(data, "$..B4") 
B5 = jsonpath(data, "$..B5") 
B6 = jsonpath(data, "$..B6") 
B7 = jsonpath(data, "$..B7") 
B8 = jsonpath(data, "$..B8") 
B8A = jsonpath(data, "$..B8A") 
B11 = jsonpath(data, "$..B11") 
B12 = jsonpath(data, "$..B12") 
B2_1 = jsonpath(data, "$..B2_1") 
B3_1 = jsonpath(data, "$..B3_1") 
B4_1 = jsonpath(data, "$..B4_1") 
B5_1 = jsonpath(data, "$..B5_1") 
B6_1 = jsonpath(data, "$..B6_1") 
B7_1 = jsonpath(data, "$..B7_1") 
B8_1 = jsonpath(data, "$..B8_1") 
B8A_1 = jsonpath(data, "$..B8A_1") 
B11_1 = jsonpath(data, "$..B11_1") 
B12_1 = jsonpath(data, "$..B12_1") 
B2_2 = jsonpath(data, "$..B2_2") 
B3_2 = jsonpath(data, "$..B3_2") 
B4_2 = jsonpath(data, "$..B4_2") 
B5_2 = jsonpath(data, "$..B5_2") 
B6_2 = jsonpath(data, "$..B6_2") 
B7_2 = jsonpath(data, "$..B7_2") 
B8_2 = jsonpath(data, "$..B8_2") 
B8A_2 = jsonpath(data, "$..B8A_2") 
B11_2 = jsonpath(data, "$..B11_2") 
B12_2 = jsonpath(data, "$..B12_2") 
B2_3 = jsonpath(data, "$..B2_3") 
B3_3 = jsonpath(data, "$..B3_3") 
B4_3 = jsonpath(data, "$..B4_3") 
B5_3 = jsonpath(data, "$..B5_3") 
B6_3 = jsonpath(data, "$..B6_3") 
B7_3 = jsonpath(data, "$..B7_3") 
B8_3 = jsonpath(data, "$..B8_3") 
B8A_3 = jsonpath(data, "$..B8A_3") 
B11_3 = jsonpath(data, "$..B11_3") 
B12_3 = jsonpath(data, "$..B12_3") 


# In[14]:


number_samples = np.size(B8,0)
dataset_spring= np.zeros((number_samples,   kernel,kernel,10), dtype=float)
dataset_summer= np.zeros((number_samples,   kernel,kernel,10), dtype=float)
dataset_autumn= np.zeros((number_samples,   kernel,kernel,10), dtype=float)
dataset_winter= np.zeros((number_samples,   kernel,kernel,10), dtype=float)
dataset= np.zeros((number_samples,   kernel,kernel,bands), dtype=float)
for i in range(0,number_samples-1):
      dataset_spring[i]=np.dstack((np.array(B2[i]),np.array(B3[i]),np.array(B4[i]),np.array(B5[i]),np.array(B6[i]),np.array(B7[i]),np.array(B8[i]),np.array(B8A[i]),np.array(B11[i]),np.array(B12[i])))
      dataset_summer[i]=np.dstack((np.array(B2_1[i]),np.array(B3_1[i]),np.array(B4_1[i]),np.array(B5_1[i]),np.array(B6_1[i]),np.array(B7_1[i]),np.array(B8_1[i]),np.array(B8A_1[i]),np.array(B11_1[i]),np.array(B12_1[i])))
      dataset_autumn[i]=np.dstack((np.array(B2_2[i]),np.array(B3_2[i]),np.array(B4_2[i]),np.array(B5_2[i]),np.array(B6_2[i]),np.array(B7_2[i]),np.array(B8_2[i]),np.array(B8A_2[i]),np.array(B11_2[i]),np.array(B12_2[i])))
      dataset_winter[i]=np.dstack((np.array(B2_3[i]),np.array(B3_3[i]),np.array(B4_3[i]),np.array(B5_3[i]),np.array(B6_3[i]),np.array(B7_3[i]),np.array(B8_3[i]),np.array(B8A_3[i]),np.array(B11_3[i]),np.array(B12_3[i])))
      dataset[i]=np.dstack((np.array(dataset_spring[i]),np.array(dataset_summer[i]),np.array(dataset_autumn[i]),np.array(dataset_winter[i])))
print(np.array(dataset).shape)
    


# In[15]:


def navie_sample(src_image, label):
    # forest-1, debris-2, water-3
    classes = label
    # get the feature space from drone image
    #with rasterio.open(src_image) as src_ds:
    #    src = src_ds.read()
       
    yield (src_image, label)


# In[16]:


########### define the result train/valid file #############
### Here the tiles are then saved into numpy arraies
root = r"/Users/siruiwang/Documents/ESPACE-LECTURE/3rd_semester/Data science of earth observation/Project_data_science/data0113/npy/"

# training samples
t_root = root + tree_type + "_samples.npy"
t_sample = []
num_sam = dataset.shape[0]
for i in range(0,num_sam-1):
    result = list(navie_sample(dataset[i], index))
    t_sample.append(result)
t_sample_array = np.array(t_sample, dtype=object)
print(t_sample_array.shape)
np.save(t_root, t_sample_array)

