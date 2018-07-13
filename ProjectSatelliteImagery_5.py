
# coding: utf-8

# # Advanced Machine Learning and Artificial Intelligence (MScA 32017)
# 
# # Project: Satellite Imagery Feature Detection
# 
# ## Notebook 3: Satellite Image Segmentation Using Simple U-Net
# 
# ### Yuri Balasanov, Mihail Tselishchev, &copy; iLykei 2018
# 
# ##### Main text: Hands-On Machine Learning with Scikit-Learn and TensorFlow, Aurelien Geron, &copy; Aurelien Geron 2017, O'Reilly Media, Inc
# 
# 
# U-Net shown below was originally proposed by [Olaf Ronneberger, Philipp Fischer and Thomas Brox](https://arxiv.org/abs/1505.04597) for biomedical image segmentation. <br>
# 
# ![unet_schema](https://ilykei.com/api/fileProxy/documents%2FAdvanced%20Machine%20Learning%2FSatellite%20Image%20Segmentation%2Funet_schema.png)
# 
# 
# This architecture consists of two parts:
# 
# 1. Contracting path (down path of the U-shape): repeated convolution/max pooling cycles, such that each $3 \times 3$ unpadded convolution - followed by ReLU - **doubles the number of channels** by applying filters, and each $2 \times 2$-stride 2 max pooling **shrinks image by 50%**
# 2. Expansive path (up path of the U-shape): repeated cycles of $2 \times 2$ up-convolution, each of them **reduces number of feature map channels by 50%**. During the up-move throught the net each output of up-convolution is concatenated with the correspondingly cropped feature map from the contracting path. Up-convolution and concatenation is followed by 2 $3 \times 3$ convolutions with ReLU
# 3. The final layer is a $1 \times 1$ convolution mapping each 64-component feature vector to probabilities of the necessary number of classes.
# 
# ##### The total number of convolutional and u-convolutional layers in the architecture is 23.
# 
# For this notebook construct and train a not so deep U-Net using Keras to illustrate how this type of architecture can be used for satellite image segmentation. 
# 
# The simplified architecture has one convolution, followed by one maxpool, followed by another convolution, and then up-convolution concatenated with the output of the first convolution, third convolution (all with ReLU activation) and final 1x1 convolution layer with sigmoid activation (see plot below). 
# 
# Note that input images have multiple channels. <br>
# Since we have 5 disjoint classes for every pixel, output map should also have multiple channels: 5.
# 
# + the horizontal blue arrow is convolution and you can see the drop in size of the image, think padding. THe number of channels keep in increasing
# + Downward arrow is max pooling, and we double the number of channels. The number of rows and cols are decreased
# + Refer the arrows in the diagram these are explained there

# In[1]:

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.optimizers import Adam
import pandas as pd
from tifffile import imsave
import tifffile as tiff



# Create a function generating network with given architecture. 

# In[2]:

# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K


def unet_model(n_classes=5, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.3, 0.1, 0.1, 0.3]):
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    n_filters *= growth_factor
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    n_filters *= growth_factor
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

    n_filters //= growth_factor
    if upconv:
        up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
    return model


#if __name__ == '__main__':
#    model = unet_model()
#    print(model.summary())
#    plot_model(model, to_file='unet_model.png', show_shapes=True)


# It is recommended not to feed whole satellite images to the network. <br>
# Instead, resample small patches of size $160 \times 160$ and train the model with such samples. <br>
# This is a common practice in image segmentation.
# 
# Note that for illustrative purposes this example uses a simpler shallow version of U-Net architecture with small number of convolutional filters.
# 
# Create U-Net model using function `simple_unet_model()` and see its summary:

# The output is created with 5 channels - one per segmentation class. Each channel will contain probabilities of pixel belonging to the corresponding class.
# 
# Save and show the neural network architecture:

# In[2]:

def normalize(img):
    min = img.min()
    max = img.max()
    return 2.0 * (img - min) / (max - min) - 1.0


# The following function takes: 
# - image *'x'* 
# - trained model 
# - patch size 
# - number of classes 
# and returns predicted probabilities of each class for every pixel of *'x'* in array with shape **(extended_height, extended_width, n_classes)**, where *'extended_height'* and *'extended_width'* are extended dimensions of *'x'* that make whole number of patches in the image.

# In[3]:

import numpy as np
import math

def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]

    # make extended img so that it contains integer number of patches
    #here we are trying to get the BB from the localization parameters which we have, look at the previous script
    npatches_vertical = math.ceil(img_height/patch_sz)
    npatches_horizontal = math.ceil(img_width/patch_sz)
    # the below code is being used since the division of the image might not lead to an integer value
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirror reflections of neighbors:
    #basically we have created an extended image and the np array contains zeros, now we are filling those zeros
    #with the original image values in terms of height and wight,
    # ask if in the extended part of the image there will be only zeros in the lower - right corner
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        # why are we filling the extended image with mirror reflections of neighbors? adn can any other technique be 
        #applied here? and why cant we just have zeros here? In case of zero the image will be either perfectly black
        #or white and should not ideally create any issue in our model
        ext_x[i, :, :] = ext_x[2*img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2*img_width - j - 1, :]

    # now assemble all patches in one array
    #so after the above processing has been done are we trying to now divide the image into a patch size of 160 and
    #then keep in appending it in form of a list and then will convert into an array in order to feed into our model?
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


# ### batch size: 
# The usage of this variable here means that how many image patches or how many number of samples needs to be shown to the network before performing a weight updation. Same goes for model.predict as well (*but will this mean that weight updation is happening even during prediction*). 
# + it may be desirable to have a different batch size when fitting the network to training data than when making predictions on test data or new input data.
# + This does become a problem when you wish to make fewer predictions than the batch size. For example, you may get the best results with a large batch size, but are required to make predictions for one observation at a time on something like a time series or sequence problem
# 
# 
# #### refer this tutorial to solve this problem if your model is facing this issue or requires different batch sizes, or whatever. https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/

# Show image of the created mask. <br>
# On this image use color codes of the first 5 colors for the 5 classes. <br>
# Create function that takes a mask created by *'predict()'* and a threshold and returns an RGB file that can be shown by *'imshow()'*. <br>
# 
# In function *'picture_from_mask()'* created below:
# - Dictionary variable *'colors'* contains first 5 colors corresponding to the 5 classes of objects. Color of each class is defined as combination of 3 basic colors
# - Dictionary *'z_order'* creates special order of classes in which the mask-image is created. If the same pixel has high enough probability of belonging to several classes then the pixel is marked as highest of them in *'z_order'*. Basically, this means that in the loop over *'z_order'* color of the next significant class replaces the color of the previous one.
# - A class of a pixel is considered "significant" if probability of that class is greater than "threshold".

# In[4]:

import random

def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    # these are just basic checkpoints, that the image needs to have 3 dimensions, the number of rows and columns 
    #withing the image need to be greater than 160, which is our patch siZE. So, if this condition is not met then 
    #ideally python should throw an error.
    assert len(img.shape) == 3 and img.shape[0] > sz     and img.shape[1] > sz     and img.shape[0:2] == mask.shape[0:2]
    
    xc = random.randint(0, img.shape[0] - sz)
    # this will generate a psuedorandom integer number between 0 adn the other number
    yc = random.randint(0, img.shape[1] - sz)
    # so this is just taking a number and starts to create a patch from that row number, but would not this lead to
    #duplicay or rather two patches having certain similar components? (by that I mean rows and columns)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]
    return patch_img, patch_mask


def get_patches(img, mask, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        # is there a possibility that due to this piece of script there could be certain parts of an image/within an
        #image be present which will never pass thorugh the model?
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)  # keras needs numpy arrays rather than lists


# In[5]:

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard


# img_t = img_normalized.transpose([1,2,0])  # keras uses last dimension for channels by default
# predicted_mask = predict(img_t, model).transpose([2,0,1])  # channels first to plot
# y_pict_2 = picture_from_mask(predicted_mask, threshold = 0.5)
# tiff.imshow(y_pict_2)

# y_pict_2.shape

# Still not perfect, but it is definitely better than it was before training and it is a good starting point.

# Do not forget to view TensorBoard graphs by calling from Terminal:
# 
# *tensorboard --logdir=tensorboard_simple_unet*
# 
# and then navigate in browser to *'localhost:6006'*.

# I need to create an array for entire X and Y, pixel by pixel and then feed it into the network for training purposes

# In[6]:

x_train_array = []
y_train_array = []
x_val_array = []
y_val_array = []


# In[9]:

def loading_images():
    n=24
    #-1 in order to remove the test image
    #assigning the X and Y data sets for training and validation
    #reading the images one by one as per the below loop and
    for i in range(1,n):
        #reading in the images
        id_img=i
        img = tiff.imread('data/mband/{}.tif'.format(id_img))
        mask = tiff.imread('data/gt_mband/{}.tif'.format(id_img)) / 255  
        # read mask and normalize it
        #normalizing the input image
        img_normalized = normalize(img)
        #dividing the image as per the below code into test and train sets
        train_xsz = int(3/4 * img.shape[1])
        # Making channels dimension last in the shape for Keras
        # For train
        img_train = img_normalized[:, :train_xsz, :].transpose([1, 2, 0])
        mask_train = mask[:, :train_xsz, :].transpose([1, 2, 0])
        # For test
        img_validation = img_normalized[:, train_xsz:, :].transpose([1, 2, 0])
        mask_validation = mask[:, train_xsz:, :].transpose([1, 2, 0])
        #creating the masks and the final training and testing sample
        TRAIN_SZ = 40  # train size, change to 4000 before running on GPU
        x_train, y_train = get_patches(img_train, mask_train, n_patches=TRAIN_SZ, sz=160)
        VAL_SZ = 10     # validation size, change to 1000 before running on GPU
        x_val, y_val = get_patches(img_validation, mask_validation, n_patches=VAL_SZ, sz=160)
        #ideally the x_val and the y_val and these variables are going to be returned as numpy arrays, so check if you 
        #still need to convert them into np array after appending
        #now we need to append all the training and validation data sets obtained from all the files from the directory
        #np.append(x_train_array, x_train, axis=0)
        x_train_array.append(x_train)
        y_train_array.append(y_train)
        x_val_array.append(x_val)
        y_val_array.append(y_val)
     
    
    #print(input_x.shape, input_y.shape, test_input_x.shape, test_input_y.shape)
    return x_train_array, y_train_array, x_val_array, y_val_array


# In[10]:

loading_images()


# In[11]:

input_x = np.array(x_train_array)
input_y = np.array(y_train_array)
test_input_x = np.array(x_val_array)
test_input_y = np.array(y_val_array)


# In[12]:

#check why this is working right now
input_x.shape


# In[13]:

input_x1= np.reshape(input_x, (input_x.shape[1]*input_x.shape[0],input_x.shape[2],input_x.shape[3],input_x.shape[4]))
input_y1 = np.reshape(input_y, (input_y.shape[1]*input_y.shape[0],input_y.shape[2],input_y.shape[3],input_y.shape[4]))
test_input_x1 = np.reshape(test_input_x, (test_input_x.shape[1]*test_input_x.shape[0],                                          test_input_x.shape[2],test_input_x.shape[3],test_input_x.shape[4]))
test_input_y1 = np.reshape(test_input_y, (test_input_y.shape[1]*test_input_y.shape[0],                                          test_input_y.shape[2],test_input_y.shape[3],test_input_y.shape[4]))


# Checking the result on the test image

# In[ ]:

test_img = tiff.imread('data/test.tif')


# The model required some amount of data augmentation for improvement, the best way to go about it is to use different kinds of rotation in the model. For this task, I will try to create two arrays of 4 channels, rotate them along with theri masks and then put them back together. 

# ### Things to do in order to increase the efficiency of the model and get a better prediction score
# + Introduce batch normalization for preventing exploding and vanishing gradients issue
# + Use dropout layers to reduce overfitting
# + Creating a deeper network
# + Rotate the images for data augmentation (you can use get_patches.py for this exercise)
# + Improve the predict function (which is used for predicting the test image), It is well-known that U-Net predicts better in the center of the patch and worse on patch borders. So, it makes sence to cut test image into overlapped patches to be sure that every region is predicted with some patch where it lies in the center
# + Create an ensemble of different networks

# ### Working on Data Augmentation

# In[14]:

# creating a function for data augmentation
def im_augmentation(x_array, y_array, x_val_array, y_val_array):
    #1.creating the first dimension in the reverse order
    #2. creating the second dimension in the reverse order
    #3. rotating the two middle dimensions by 90 degrees
    #for the x_array
    x1 = x_array[::-1,:,:]
    x2 = x_array[:,::-1,:]
    x3 = np.rot90(x_array, 2)
    #for the y labels
    y1 = y_array[::-1,:,:]
    y2 = y_array[:,::-1,:]
    y3 = np.rot90(y_array, 2)
    #for the x_validation array
    x_val1 = x_val_array[::-1,:,:]
    x_val2 = x_val_array[:,::-1,:]
    x_val3 = np.rot90(x_val_array, 2)
    #for the y_validation array
    y_val1 = y_val_array[::-1,:,:]
    y_val2 = y_val_array[:,::-1,:]
    y_val3 = np.rot90(y_val_array, 2)
    # in the end putting all the arrays together and obtaining the output
    global x_aug_input
    global y_aug_input
    global x_val_aug_input
    global y_val_aug_input
    x_aug_input = np.vstack((x_array, x1, x2, x3))
    y_aug_input = np.vstack((y_array, y1, y2, y3))
    x_val_aug_input = np.vstack((x_val_array, x_val1, x_val2, x_val3))
    y_val_aug_input = np.vstack((y_val_array, y_val1, y_val2, y_val3))
    print ("old_shape", x_array.shape, y_array.shape, x_val_array.shape, y_val_array.shape)
    print("new_shape", x_aug_input.shape, y_aug_input.shape, x_val_aug_input.shape, y_val_aug_input.shape)
    print("you need to check that the 1st dimension has values 4 times that of the input, because you are    using 3 arrays of augmented data on top of the initial one")
    return x_aug_input, y_aug_input, x_val_aug_input, y_val_aug_input


# In[15]:

im_augmentation(input_x1, input_y1, test_input_x1, test_input_y1)


# ### Feeding the augmented data set into the model

# In[30]:

# Now training the model:
N_EPOCHS = 100
BATCH_SIZE = 100
# ask Keras to save best weights (in terms of validation loss) into file:
model_checkpoint = ModelCheckpoint(filepath='weights_simple_unet_2.hdf5', monitor='val_loss', save_best_only=True)
# ask Keras to log each epoch loss:
csv_logger = CSVLogger('log_2.csv', append=True, separator=';')
# ask Keras to log info in TensorBoard format:, but right now we dont need to check the TF graph
#tensorboard = TensorBoard(log_dir='tensorboard_simple_unet/', write_graph=True, write_images=True)
# Fit:
np.random.seed(1)
model.fit(x_aug_input, y_aug_input, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          verbose=2, shuffle=True,
          callbacks=[model_checkpoint, csv_logger],
          validation_data=(x_val_aug_input, y_val_aug_input))


# In[ ]:

test_img_normalized = normalize(test_img)
test_img_t = test_img_normalized.transpose([1,2,0])  # keras uses last dimension for channels by default
predicted_mask = predict(test_img_t, model).transpose([2,0,1])  # channels first to plot
y_pict_2 = picture_from_mask(predicted_mask, threshold = 0.5)
tiff.imshow(y_pict_2)


# In[ ]:

tiff.imsave('predicted_mask.tif', (255*predicted_mask).astype('uint8'))
tiff.imsave('y_pict_2.tif', y_pict_2)

