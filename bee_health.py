# Simple CNN for the classification of images centered on individual bees into
# binary classes healthy (1) and non-healthy (0)

# The health label in this dataset has six classes, one for healthy bees and 
# five for bees in various conditions of distress, mostly due to some invasive 
# insect to their hive. However, this project will simplify the health label to
# a binary class, healthy and non-healthy, since it may later be interesting to 
# see how well such a binary model would generalize to images of other types of 
# non-healthy bees (ex. foulbrood, fungal disease, pesticide exposure, 
# colony collapse disorder). Once images of bees with these other health issues
# is assembled, transfer learning could also be used.  

# Amazing dataset provided by Jenny Yang on kaggle
# https://www.kaggle.com/jenny18/honey-bee-annotated-images 
# (CC0: Public Domain)

### 1. Imports
import pandas as pd
import numpy as np
import skimage
import skimage.io
import skimage.transform

from sklearn.model_selection import train_test_split

from tensorflow import keras

### 2. Load and inspect data
# non-image data
df_raw = pd.read_csv('./honey-bee-annotated-images/bee_data.csv')
df_raw.info()
# 5172 rows/examples and 9 columns/features
# no missing data (on non-image data, need to check all images also present)
print('available features:\n', df_raw.columns.values)
# 9 columns: 'file' 'date' 'location' 'zip code' 'subspecies' 'health'
#            'pollen_carrying' 'caste'
# label y: 'health'
df_raw['health'].value_counts()
# classes (no. examples): healthy (3384)
#                         few varrao, hive beetles (579)
#                         Varrao, Small Hive Beetles (472)
#                         ant problems (457)
#                         hive being robbed (251)
#                         missing queen (29)    
# labeled classes are mutually exlusive (not multiclass)
# will simplify this to a binary healthy vs non-healthy
# distribution will be 3384 healthy and 1788 non-healthy                     

# image data
img_folder = './honey-bee-annotated-images/bee_imgs/bee_imgs/'
# .png files with side lengths ranging from about 50 to 200 pixels
# no missing images, all named to match a value in the file column of df_raw

### 3. Preprocess label data - y
# one hot encode health, isolate one binary for healthy (1) vs non-healthy (0)
df = pd.get_dummies(df_raw, columns=['health'])
y = df.health_healthy
files_X = df.file
# split into train (64%), dev (16%), and test (20%) sets
temp_X, files_X_test, temp_y, y_test = train_test_split(files_X, y, 
                                                        test_size=0.2, 
                                                        random_state=123, 
                                                        stratify=y)
files_X_train, files_X_dev, y_train, y_dev = train_test_split(temp_X, temp_y, 
                                                              test_size=0.2, 
                                                              random_state=456, 
                                                              stratify=temp_y)
# (mtrain=3309, mdev=828,, mtest=1035)

### 4. Preprocess feature data (images) - X
# function to load, resize, and group images into their sets
# >>> definitely a faster way to do this, will revisit <<<
resize_height=64 
resize_width=64
def img_grouping(file_group_raw):
    num_files = file_group_raw.size
    file_group = file_group_raw.reset_index(drop=True)
    for i in range(0, num_files):
        img_raw = skimage.io.imread(img_folder + file_group[i])
        img = skimage.transform.resize(img_raw, (resize_width, resize_height),
                                       mode='reflect')
        # seems images are inconsistent having the RGBA fourth alpha channel, 
        # drop it and stick with RGB
        img = img[:, :, :3] 
        if i==0:
            img_group = np.concatenate(([[img]]), axis=0)
        else:
            img_group = np.concatenate((img_group, [img]), axis=0)
    return img_group
X_train = img_grouping(files_X_train)
X_dev = img_grouping(files_X_dev)
X_test = img_grouping(files_X_test)

### 5. First CNN attempt
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=5, kernel_size=(5,5), padding='valid', 
                              activation='relu',
                              kernel_regularizer=keras.regularizers.l2(0.01),
                              input_shape=(resize_width, resize_height, 3)))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))
model.add(keras.layers.Conv2D(filters=8, kernel_size=(5,5), padding='valid', 
                              activation='relu',
                              kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(filters=10, kernel_size=(5,5), padding='valid',
                              activation='relu',
                              kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(50, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Dense(10, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(keras.layers.Dense(1, activation='sigmoid',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=50)
# training performance: loss=0.3461 and acc=0.9275

# compare to dev set
dev_loss, dev_acc = model.evaluate(X_dev, y_dev)
print('dev set: loss=%0.4f acc=%0.4f' % (dev_loss, dev_acc)) 
# dev performance: loss=0.3546 and acc=0.9251

# Training and dev performances are very similar, so will not worry about 
# variance for the moment. Will later attempt to increase accuracy with some
# changes to the network architecture and other hyperparameters. Will continue
# using dev set performance to compare these new models, and will reserve test
# set for a measure of quality once one final model is chosen. 

### 6. Exports
# export current model
model.save('bee_health_model1.h5')









