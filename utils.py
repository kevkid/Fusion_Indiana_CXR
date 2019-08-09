#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:37:38 2018

@author: kevin
Utils file
Place all common methods here
"""
import os
import get_images_from_db
import cv2
from keras.models import Sequential
from keras.layers import Dense, Reshape, concatenate, Lambda, Average, Maximum, Add, Multiply, Concatenate, BatchNormalization, Activation, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Dropout, Flatten
from keras.preprocessing.sequence import pad_sequences
from itertools import islice
import itertools
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
from pycocotools.coco import COCO
import pylab
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
import matplotlib.pyplot as plt
import collections

def get_data(csv_fname = 'image_list.csv',from_db = False, classes = list(range(0,23)), uniform = True):
    if from_db:
        image_list = get_images_from_db.get_images_from_db("labeled")
        image_list.to_csv(csv_fname)
    else:
        image_list = pd.read_csv(csv_fname)
    #get classes from input
    image_list_subset = image_list[image_list.class_id.isin(classes)]
    if uniform:
        image_list_subset = image_list_subset.groupby('class_id').head(
                min(image_list_subset.class_id.value_counts()))
    return image_list_subset

#TODO: Make the images load using relative location
def get_images_from_df(df, shape, loc_col="location"):
    images = []
    for i, (index, sample) in enumerate(df.iterrows()):
            #read image
            image = cv2.cvtColor(cv2.imread(sample[loc_col]), cv2.COLOR_BGR2RGB)#Colors were in BGR instead of RGB
            image = cv2.resize(image, shape[:2])
            if len(image.shape) < 3:#if there is no channel data, just give it 3 channels
                image = np.stack((image,)*3, -1)
            images.append(image)
    return np.array(images)
#TODO: Make the images load using relative location
def get_images_from_df_one_channel(df, shape):
    images = []
    for i, (index, sample) in enumerate(df.iterrows()):
            #read image
            image = cv2.cvtColor(cv2.imread(sample["location"]), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, shape)
            if len(image.shape) == 3:#if there is no channel data
                image = image[:,:,1]
            images.append(image)
    return np.array(images)

def caption2sequence(captions, tokenizer, max_text_len = 200, padding = 'post'):
    for i, caption in enumerate(captions):
        sequence = tokenizer.texts_to_sequences([caption])    
        padded_seq = pad_sequences(sequence, maxlen=max_text_len, padding=padding)
        captions[i] = padded_seq.flatten()
    return captions

def take(n, iterable):#vocabulary size MUST be smaller than vocab generated from text
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))
#TODO: Fix ncomponents
def load_glove(GLOVE_DIR, word_index, tokenizer, vocabulary_size = 10000, n_components = 106, EMBEDDING_DIM=300):
    from sklearn.decomposition import PCA
    #load embeddings
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    print(np.shape(embedding_matrix))
    print('performing PCA')
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embedding_matrix)
    return pca_result

def mapTokensToEmbedding(embedding, word_index, vocabulary_size):#function to map embeddings from my w2v
    embedding_matrix = np.zeros((vocabulary_size, len(embedding['-UNK-'])))
    for word, index in word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embedding.get(word)#Takes index of word 
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix
'''
    Flattening means to take the samples and extract the sentences regardless 
    of structure of the samples
'''
def filter_sentences(documents, flatten = True):
    print('filtering sentences')
    if flatten:
        sents = [nltk.sent_tokenize(s) for s in documents]
        sents = list(itertools.chain.from_iterable(sents))
    else:
        sents = documents
    sents = [x.strip() for x in sents]
    print('filtering sents and removing stopwords')
    filtered_sents = []
    import re
    stop_words = set([ "a", "about", "above", "after", "again", "against", 
                  "all", "am", "an", "and", "any", "are", "as", "at", "be", 
                  "because", "been", "before", "being", "below", "between", 
                  "both", "but", "by", "could", "did", "do", "does", "doing",
                  "down", "during", "each", "few", "for", "from", "further", 
                  "had", "has", "have", "having", "he", "he'd", "he'll", "he's", 
                  "her", "here", "here's", "hers", "herself", "him", "himself", 
                  "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", 
                  "in", "into", "is", "it", "it's", "its", "itself", "let's",
                  "me", "more", "most", "my", "myself", "nor", "of", "on", 
                  "once", "only", "or", "other", "ought", "our", "ours", 
                  "ourselves", "out", "over", "own", "same", "she", "she'd", 
                  "she'll", "she's", "should", "so", "some", "such", "than", 
                  "that", "that's", "the", "their", "theirs", "them",
                  "themselves", "then", "there", "there's", "these", "they", 
                  "they'd", "they'll", "they're", "they've", "this", "those", 
                  "through", "to", "too", "under", "until", "up", "very", 
                  "was", "we", "we'd", "we'll", "we're", "we've", "were", 
                  "what", "what's", "when", "when's", "where", "where's", 
                  "which", "while", "who", "who's", "whom", "why", "why's", 
                  "with", "would", "you", "you'd", "you'll", "you're", 
                  "you've", "your", "yours", "yourself", "yourselves", "ing" ] + 
                    stopwords.words('english'))
    stop_words_re = re.compile(r'\b(?:%s)\b' % '|'.join(stop_words))
    for sent in sents:
        s = sent.lower()
        s = " ".join(re.findall("[a-zA-Z-]+", s))
        s = re.sub(stop_words_re, '', s)
        s = re.sub(' +',' ',s)
        s = re.sub('\'s','',s)
        #s = " ".join(nltk.PorterStemmer().stem(x) for x in s.split())#may not need to stem...
#            if len(s.split()) > 2:
        filtered_sents.append(s)
    return filtered_sents

def load_PMC(csv_fname, classes, uniform = True):
    return get_data(csv_fname=csv_fname, classes=classes, uniform=uniform)
#TODO use different subsets for train and test
def load_COCO(location, class_names, dataType='train2017'):
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    #####################Get Training data
    dataDir=location#'/media/kevin/1142-5B72/coco_dataset'
    #dataDir='/home/kevin/Documents/Lab/coco_dataset'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    #I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)
    #encapsulate this function
    def build_df_coco(imgids, class_ID, cat_id):
        ImgIdsDict = {}
        for imgid in imgids:
            annIds = coco_caps.getAnnIds(imgIds=imgid);
            captions = [ann['caption'] for ann in coco_caps.loadAnns(annIds)]
            
            annIdsBbox = coco.getAnnIds(imgIds=imgid, catIds=cat_id)
            bbox = [ann['bbox'] for ann in coco.loadAnns(annIdsBbox)]
            img = coco.loadImgs(imgid)[0]
            ImgIdsDict[imgid] = {
                    'class_id':class_ID,
                    'caption':' '.join(captions),
                    'location': '%s/images/%s/%s'%(dataDir,dataType,img['file_name']),
                    'bbox': [set_bbox_to_ratio(x, (img['width'], img['height'])) for x in bbox],
                    'shape': (img['width'], img['height'])
                    }
        df = pd.DataFrame.from_dict(ImgIdsDict, orient='index')
        return df
    def set_bbox_to_ratio(bbox, shape):
        x=bbox[0]/shape[0]
        y=bbox[1]/shape[1]
        width=bbox[2]/shape[0]
        height=bbox[3]/shape[1]
        return [x,y,width,height]
    #get images of dogs:
    catIds = coco.getCatIds(catNms=class_names);#intersection of images with all categories
    img_ids = {x:coco.getImgIds(catIds=x) for x in catIds}#gets our image ids
    n = {a:[c for c in b if not any(c in h for j, h in img_ids.items() if j != a)] for a, b in img_ids.items()}
    filtered_ids_dfs = []
    for i, (k,v) in enumerate(n.items()):
        filtered_ids_dfs.append(build_df_coco(v, i, k))
    data_df = pd.concat(filtered_ids_dfs)
    return data_df

def load_indiana_CXR(path = '/gpfs/ysm/pi/krauthammer/kl533/Indiana_University_Chest_X-ray_Collection'):
    import pandas as pd
    from os import listdir
    from collections import OrderedDict
    from xml.dom import minidom
    import json
    from xml.etree.ElementTree import fromstring
    from xmljson import Parker, BadgerFish
    import matplotlib.pyplot as plt
    path_i = path + '/images'
    path_r = path + '/reports'
    '''
    For each image save its id and store the freetext, labels.
    Since each record can have several images we get each image, store its id, get the 
    list of labels, and whole freetext and store it
    '''
    def get_reports():
        bf = BadgerFish(dict_type=dict)
        reports = []
        for file in listdir(path_r):
            file = path_r+'/'+file
            with open(file, "r") as xml:
                reports.append(bf.data(fromstring(xml.read())))
        return reports
    def get_text(report):
        sample = {}
        tmp_txt = ''
        #Get freetext
        for label in report['eCitation']['MedlineCitation']['Article']['Abstract']['AbstractText']:
            try:#sometimes we don't have a freetext label
                tmp_txt += "{}: {}\n".format(label['@Label'], label['$'])
            except:
                print('missing some free text labels')
        sample['radiology_report'] = tmp_txt
        #get labels
        if isinstance(report['eCitation']['MeSH']['major'],list):
            labels = []
            for label in report['eCitation']['MeSH']['major']:
                labels.append(label['$'])
            sample['labels'] = labels
        else:
            sample['labels'] = [report['eCitation']['MeSH']['major']['$']]
        return sample
    reports = get_reports()
    samples = []
    for report in reports:
        if 'parentImage' in report['eCitation']:
            #if multiple images
            if isinstance(report['eCitation']['parentImage'], list):
                for image in report['eCitation']['parentImage']:
                    sample = {}
                    if image['@id'] != '':
                        #print('loaded {}'.format(image['@id']))
                        sample['img_id'] = image['@id']
                        sample.update(get_text(report))#pass in the report
                    samples.append(sample)
            #if just one image        
            else:
                image = report['eCitation']['parentImage']
                sample = {}
                if image['@id'] != '':
                    print('loaded {}'.format(image['@id']))
                    sample['img_id'] = image['@id']
                    sample.update(get_text(report))#pass in the report
                samples.append(sample)
    return pd.DataFrame(samples)

#We are automatically normalizing image by dividing by 255
def get_model_data(data, seed = 0, test_size = 0.2, vocabulary_size = 10000, filters = '', max_text_len=200, shape = (106,106), split = False, 
                   text_col = 'caption', label_col = 'class_id', loc_col='location'):
    onehotencoder = OneHotEncoder()
    if split == False:
        train, test = train_test_split(data, test_size=test_size, random_state=seed)
        #set up tokenizer
        tokenizer = Tokenizer(num_words= vocabulary_size, lower=True, filters=filters)
        tokenizer.fit_on_texts(filter_sentences(data[text_col]))
        #get our Training and testing set ready
        X_text_train = caption2sequence(filter_sentences(train[text_col].copy(), flatten=False), tokenizer, max_text_len)
        X_text_test = caption2sequence(filter_sentences(test[text_col].copy(), flatten=False), tokenizer, max_text_len)    
        X_image_train = get_images_from_df(train, shape)/255
        X_image_test = get_images_from_df(test, shape)/255
        y_train = np.array(train[label_col]).reshape((-1, 1))
        y_test = np.array(test[label_col]).reshape((-1, 1))
        y_train = onehotencoder.fit_transform(y_train).toarray()
        y_test = onehotencoder.transform(y_test).toarray()
        return (X_text_train, X_text_test, X_image_train, X_image_test, y_train, y_test, tokenizer)
    else:
        #set up tokenizer
        tokenizer = Tokenizer(num_words= vocabulary_size, lower=True, filters=filters)
        tokenizer.fit_on_texts(filter_sentences(data[text_col]))
        #get our Training and testing set ready
        X_text = caption2sequence(filter_sentences(data[text_col].copy(), flatten=False), tokenizer, max_text_len)
        X_image = get_images_from_df(data, shape, loc_col=loc_col)/255
        y = np.array(data[label_col]).reshape((-1, 1))
        y = onehotencoder.fit_transform(y).toarray()
        return (X_text, X_image, y, tokenizer)

def embedding_3Ch(inputs):
    return K.stack((inputs,)*3, -1)

def image_3Ch_to_1Ch(inputs, shape):#Should be sending in a 100x100x3px image to convert it into a 300x100 image
    return K.reshape(inputs, shape)

    '''Returns our predictions'''
def test_model(model, data, image = True, text = True, class_names = None):
    if image and text:
        X_text_test, X_image_test, y_test = zip(*data)
        X_image_test = list(X_image_test)
        X_text_test = list(X_text_test)
        y_test = list(y_test)
        print(model.evaluate(x=[X_text_test, X_image_test], y=[y_test]))
        y_hat=model.predict([X_text_test, X_image_test]).argmax(axis=-1)
    elif image and not text:
        X_image_test, y_test = zip(*data)
        X_image_test = list(X_image_test)
        y_test = list(y_test)
        print(model.evaluate(x=[X_image_test], y=[y_test]))
        y_hat=model.predict([X_image_test]).argmax(axis=-1)
    elif not image and text:
        X_text_test, y_test = zip(*data)
        X_text_test = list(X_text_test)
        y_test = list(y_test)
        print(model.evaluate(x=[X_text_test], y=[y_test]))
        y_hat=model.predict([X_text_test]).argmax(axis=-1)
    
    cm = confusion_matrix([np.argmax(t) for t in y_test], y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    report = classification_report([np.argmax(t) for t in y_test], y_hat, target_names=class_names)
    print(report)
    return y_hat

#https://github.com/keras-team/keras/issues/8123
class MultiGPUCheckpoint(ModelCheckpoint):
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

def get_multi_gpu_model(model, G):
    # check to see if we are compiling using just a single GPU
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = Model(inputs=model.inputs, outputs=model.outputs)
    # otherwise, we are compiling using multiple GPUs
    else:
        print("[INFO] training with {} GPUs...".format(G))
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = Model(inputs=model.inputs, outputs=model.outputs)
        # make the model parallel
        model = multi_gpu_model(model, gpus=G)
    return model
def plot_model_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
'''Converts classification report into a pandas DataFrame'''
def classification_report_df(report):
    report_data = []
    lines = report.split('\n')
    del lines[-5]
    del lines[-1]
    del lines[1]
    for line in lines[1:]:
        row = collections.OrderedDict()
        row_data = line.split()
        row_data = list(filter(None, row_data))
        if len(row_data) > 5:#class has a space such as micro average
            row['class'] = row_data[0] + "_" + row_data[1]
            del row_data[1]
        else:
            row['class'] = row_data[0]
        
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = int(row_data[4])
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    df.set_index('class', inplace=True)
    return df
'''Calculates the mean for each row in a pandas DataFrame'''
def df_mean(df_list):
    df_concat = pd.concat(df_list)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    return df_means

'''
This allows us to print the first and last x number of images and their sample number
This is to help visualize our images when we are cross validating so we have the sames 
images for each cross validation. We want to use the same cross validation sets across models
'''
def print_samples(samples,images = None, text = None, tokenizer = None, x = 2):
    fist_last = np.append(samples[:x],samples[-x:])
    print(fist_last)
    if images is not None:
        imgs = images[fist_last]
        w=10
        h=10
        fig=plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        for i in range(len(imgs)):
            img = imgs[i]
            fig.add_subplot(rows, columns, i+1)
            plt.text(0, 0, "id: {}".format(fist_last[i]), fontsize=14, color='blue')
            plt.imshow(img)
        plt.show()
    if text is not None:
        for i in fist_last:
            txt_ = list(text[i])
            txt_ = list(filter(lambda a: a != 0, txt_))
            print("id: {}".format(i))
            [print(tokenizer.index_word[j] + " ", end='', flush=True) for j in txt_]
            print("\n=======")
'''
helper function to check if weights have changed from different keras.get_weights()
'''
def compare_weights(w1, w2):
    assert(w1)
    assert(w2)
    assert(len(w1) == len(w2))
    for i in range(len(w1)):
        eq = np.all(w1[i]==w2[i])
        if not eq:
            return False
    return True

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
        
def set_weights(model, layer_name, loc, set_layer_name = None):
    ri_weights = model.get_weights()
    if set_layer_name:
        #sets the current models sequental name to the desired name, in the event our model and weights have different names
        model.layers[getLayerIndexByName(model,layer_name)].name = set_layer_name
    model.load_weights(loc, by_name=True)
    loaded_weights = model.get_weights()
    assert(not compare_weights(ri_weights, loaded_weights))#checks if weights were loaded
#Batch norm after activation
'''
https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/dgqy1yt

"The whole purpose of the BN layer is to output zero mean and unit variance output. If you put the relu after it, you are not going to have zero mean and variance will be half too, which defies the whole purpose of putting BN at the first place. I think relu before BN makes sense by above reasoning."

https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
'''
def get_img_branch():
    Image_Branch = Sequential(name='Image_Branch')
    #block 1
    Image_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    Image_Branch.add(BatchNormalization(name="block1_bn1"))
    Image_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    Image_Branch.add(BatchNormalization(name="block1_bn2"))
    Image_Branch.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 2
    Image_Branch.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    Image_Branch.add(BatchNormalization(name="block2_bn1"))
    Image_Branch.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    Image_Branch.add(BatchNormalization(name="block2_bn2"))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 3
    Image_Branch.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv1'))
    Image_Branch.add(BatchNormalization(name="block3_bn1"))
    Image_Branch.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv2'))
    Image_Branch.add(BatchNormalization(name="block3_bn2"))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 4
    Image_Branch.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv1'))
    Image_Branch.add(BatchNormalization(name="block4_bn1"))
    Image_Branch.add(Conv2D(512, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv2'))
    Image_Branch.add(BatchNormalization(name="block4_bn2"))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #Flatten
    Image_Branch.add(Flatten())
    return Image_Branch

def get_text_branch():
    Text_Branch = Sequential(name='Text_Branch')
    #block 1
    Text_Branch.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    Text_Branch.add(BatchNormalization(name="block1_bn1"))
    Text_Branch.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    Text_Branch.add(BatchNormalization(name="block1_bn2"))
    Text_Branch.add(MaxPooling1D(pool_size=2, name='block1_pool'))
    Text_Branch.add(Dropout(0.1))
    #block 2
    Text_Branch.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    Text_Branch.add(BatchNormalization(name="block2_bn1"))
    Text_Branch.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    Text_Branch.add(BatchNormalization(name="block2_bn2"))
    Text_Branch.add(MaxPooling1D(pool_size=3, name='block2_pool'))
    Text_Branch.add(Dropout(0.1))    
    #Flatten
    Text_Branch.add(Flatten())
    return Text_Branch   

def get_text_branch_2D():
    Image_Branch = Sequential(name='Text_Branch')
    #block 1
    Image_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    Image_Branch.add(BatchNormalization(name="block1_bn1"))
    Image_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    Image_Branch.add(BatchNormalization(name="block1_bn2"))
    Image_Branch.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 2
    Image_Branch.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    Image_Branch.add(BatchNormalization(name="block2_bn1"))
    Image_Branch.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    Image_Branch.add(BatchNormalization(name="block2_bn2"))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 3
    Image_Branch.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv1'))
    Image_Branch.add(BatchNormalization(name="block3_bn1"))
    Image_Branch.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv2'))
    Image_Branch.add(BatchNormalization(name="block3_bn2"))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 4
    Image_Branch.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv1'))
    Image_Branch.add(BatchNormalization(name="block4_bn1"))
    Image_Branch.add(Conv2D(512, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv2'))
    Image_Branch.add(BatchNormalization(name="block4_bn2"))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #Flatten
    Image_Branch.add(Flatten())
    return Image_Branch

def get_text_branch_2D_small():
    Text_Branch = Sequential(name='Text_Branch')
    #block 1
    Text_Branch.add(Conv2D(32, kernel_size=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    Text_Branch.add(BatchNormalization(name="block1_bn1"))
    Text_Branch.add(Conv2D(32, kernel_size=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    Text_Branch.add(BatchNormalization(name="block1_bn2"))
    Text_Branch.add(MaxPooling2D(pool_size=(2,2), name='block1_pool'))
    Text_Branch.add(Dropout(0.1))
    #block 2
    Text_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    Text_Branch.add(BatchNormalization(name="block2_bn1"))
    Text_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    Text_Branch.add(BatchNormalization(name="block2_bn2"))
    Text_Branch.add(MaxPooling2D(pool_size=(3,3), name='block2_pool'))
    Text_Branch.add(Dropout(0.1))    
    #Flatten
    Text_Branch.add(Flatten())
    return Text_Branch          
'''
def get_CNN():
    CNN_Model = Sequential()
    #block 1
    CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    CNN_Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    CNN_Model.add(Dropout(0.2))
    
    #block 2
    CNN_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    CNN_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    CNN_Model.add(Dropout(0.2))
    
    #block 3
    CNN_Model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv1'))
    CNN_Model.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv2'))
    CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    CNN_Model.add(Dropout(0.2))
    
    #block 4
    CNN_Model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv1'))
    CNN_Model.add(Conv2D(512, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv2'))
    CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    CNN_Model.add(Dropout(0.5))
    
    #Flatten
    CNN_Model.add(Flatten())
    #Dense section
    CNN_Model.add(Dense(512, activation='relu', name='dense_layer1'))
    CNN_Model.add(Dropout(0.5))
    CNN_Model.add(Dense(512, activation='relu', name='dense_layer2'))
    CNN_Model.add(Dropout(0.5))
    return CNN_Model

def get_CNN_TEXT():
    CNN_Model = Sequential()
    #block 1
    CNN_Model.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    CNN_Model.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    CNN_Model.add(MaxPooling1D(pool_size=2, name='block1_pool'))
    CNN_Model.add(Dropout(0.2))
    #block 2
    CNN_Model.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    CNN_Model.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    CNN_Model.add(MaxPooling1D(pool_size=3, name='block2_pool'))
    CNN_Model.add(Dropout(0.2))    
    #Flatten
    CNN_Model.add(Flatten())
    #Dense section
    CNN_Model.add(Dense(1024, activation='relu', name='dense_layer1'))
    CNN_Model.add(Dropout(0.5))
    CNN_Model.add(Dense(512, activation='relu', name='dense_layer2'))
    CNN_Model.add(Dropout(0.5))
    return CNN_Model
'''