#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec Keras implementation
Created on Mon Aug 13 11:33:41 2018

@author: kevin
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.initializers import TruncatedNormal
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import itertools
import keras
import pandas as pd
import random
from utils import filter_sentences
class w2v_keras:
    
    def __init__(self, vocabulary_size = 10000, window_size = 2, 
                 epochs = 1000000, valid_size = 16, vector_dim = 106, valid_window = 100, 
                 filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890\n\r→▶≡≈†p‡', 
                 embedding_file_location = 'embeddings_weights', neg_samples = 10):
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.vector_dim = vector_dim
        self.epochs = epochs
        self.valid_size = valid_size# Random set of words to evaluate similarity on.
        self.valid_window = valid_window # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        self.filters = filters
        self.embedding_file_location = embedding_file_location
        self.neg_samples = neg_samples
        self.model_fit = False
        self.model, self.validation_model = self.__get_model()


    def __generate_skipgrams(self, documents):
        #generate skipgrams
        print('creating sents ({} rows)'.format(len(documents)))
        #sents = newsgroups_train.data
        
            
        sents = filter_sentences(documents)
        self.filtered_sents = sents
        print('tokenizing sents ({} sentences)'.format(len(sents)))
        self.tokenizer = Tokenizer(num_words= self.vocabulary_size, lower=True, filters=self.filters)
        self.tokenizer.fit_on_texts(sents)
        self.word_index_inv = {v: k for k, v in self.tokenizer.word_index.items()}
        sequences = self.tokenizer.texts_to_sequences(sents)    
        sampling_table = make_sampling_table(self.vocabulary_size, sampling_factor=0.001)
        print('generating couples')
        couples = []
        labels = []
        for seq in sequences:
            c,l = skipgrams(seq, vocabulary_size=self.vocabulary_size, 
                    window_size=self.window_size, shuffle=True, sampling_table=sampling_table, 
                    negative_samples=self.neg_samples)
            couples.extend(c)
            labels.extend(l)
        
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        return word_target, word_context, labels
    
    
    
    def __get_model(self, load_weights = False):
        stddev = 1.0 / self.vector_dim
        initializer = TruncatedNormal(mean=0.0, stddev=stddev, seed=None)
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        input_target = Input((1,))
        input_context = Input((1,))
        embedding = Embedding(self.vocabulary_size, self.vector_dim, input_length=1, name='embedding', 
                              embeddings_initializer=initializer)
        
        target = embedding(input_target)
        context = embedding(input_context)
        # setup a cosine similarity operation which will be output in a secondary model
        similarity = dot([target, context], normalize=True, axes=-1)#we can see that we don't use the similarity for anything but checking
        # now perform the dot product operation to get a similarity measure
        dot_product = dot([target, context], normalize=False, axes=-1)
        dot_product = Reshape((1,), input_shape=(1,1))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        # create the primary training model
        model = Model(inputs=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # create a secondary validation model to run our similarity checks during training
        validation_model = Model(inputs=[input_target, input_context], outputs=similarity)
        
        return model, validation_model
    
    
    def fit(self, documents = None, batch_size = 10000, epochs = 200, get_sim = True):
        if self.model_fit == False and documents is not None:#never fit before...
            word_target, word_context, labels = self.__generate_skipgrams(documents)
            df = pd.DataFrame({'word_target': word_target, 'word_context': word_context, 
                       'labels': labels})
            self.skip_grams = df
            self.model_fit = True
            if get_sim:
                similar = get_similar(self.valid_examples, self.word_index_inv, self.run_sim)
            reduce_lr = ReduceLROnPlateau(mode='auto', monitor='acc', factor=0.2, patience=2, min_lr=1e-5, verbose=1)
            history = self.model.fit([self.skip_grams['word_target'],self.skip_grams['word_context']], self.skip_grams['labels'], 
                        batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, similar])
            return history
        elif self.model_fit == True and documents is None:#continue training if the model is already fit
            if get_sim:
                similar = get_similar(self.valid_examples, self.word_index_inv, self.run_sim)
            reduce_lr = ReduceLROnPlateau(mode='auto', monitor='acc', factor=0.2, patience=2, min_lr=1e-5, verbose=1)
            history = self.model.fit([self.skip_grams['word_target'],self.skip_grams['word_context']], self.skip_grams['labels'], 
                        batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, similar])
            return history
        else:
            print('The model is fitted, but you are submitting new documents.\nYou should either continue training or start training a fresh model.')
        return None
    
    #similarity method
    def run_sim(self, word, top_k = 10):
        voc_size = min(self.vocabulary_size, len(self.tokenizer.word_index))
        if word in self.tokenizer.word_index:
            valid_word = self.tokenizer.word_index[word]
            sim = self.validation_model.predict([[valid_word]*(voc_size), 
                                             list(range(voc_size))]).flatten()
            nearest = sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)[1:top_k+1]#(-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % word
            for k in range(top_k):
                close_word = self.word_index_inv[nearest[k]]
                log_str = '%s %s %f,' % (log_str, close_word, sim[nearest[k]])
            print(log_str)
        else:
            print('key not in vocab')
    def take(self, n, iterable):#vocabulary size MUST be smaller than vocab generated from text
        "Return first n items of the iterable as a list"
        return dict(itertools.islice(iterable, n))    
    def get_embeddings(self, as_dict = True):
        if as_dict:
            embeddings = {}
            weights = self.model.layers[2].get_weights()[0]
            vocab_words = self.take(self.vocabulary_size-1, #we take -1 because our embedding[0] doesnt correspond to any word, tokenizer starts at 1
                            self.tokenizer.word_index.items())
            for word, index in vocab_words.items():
                embeddings[word] = weights[index]
                
            embeddings['-UNK-'] = weights[0]#weights for the unknown word
            return embeddings
        else:
            vocab_words = np.array(list(self.take(self.vocabulary_size, 
                            self.tokenizer.word_index.items()).keys())).reshape(-1,1)
            embeddings = np.hstack([vocab_words, self.model.layers[2].get_weights()[0]])
            return embeddings
    def get_model(self):
        return self.model
    def get_skip_grams(self):
        self.skip_grams
    def get_vocabulary(self):
        return self.tokenizer.word_index
    def get_reverse_vocabulary(self):
        return self.word_index_inv
    def save_weights(self, location):
        if self.model.built:
            self.model.save_weights(location)
        else:
            print('Model is not built!')
        return 0
    def load_weights(self, location):
        if self.model.built:
            self.model.load_weights(location)
        else:
            print('Model is not built!')
    def save_embeddings(self, location):
        import json
        if self.model.built:
            emb = self.get_embeddings()
            
            for key, val in emb.items():
                emb[key] = val.tolist()
            
            data = json.dumps(emb)
            with open(location, 'w') as outfile:
                json.dump(data, outfile)
        else:
            print('Model is not built!')
    def load_embeddings(self, embeddings = None, location = None):
        if self.model.built:
            if embeddings is not None:
                self.model.layers[2].set_weights(
                        embeddings.reshape(1, np.shape(embeddings)[0], np.shape(embeddings)[1]))
            else:
                import json
                with open(location) as f:
                    embs = json.load(f)
                    emb = json.loads(embs)
                embeddings = np.array(list(emb.values()))
                self.model.layers[2].set_weights(embeddings.reshape(1, np.shape(embeddings)[0], np.shape(embeddings)[1]))
        else:
            print('Model is not built!')
    
      
class get_similar(keras.callbacks.Callback):
    def __init__(self, valid_examples, word_index_inv, run_sim):
        self.valid_examples = valid_examples
        self.word_index_inv = word_index_inv
        self.run_sim = run_sim
    def on_epoch_end(self, epoch, logs=None):
        for ex in self.valid_examples:
            if ex > 0:
                self.run_sim(self.word_index_inv[ex], 5)
class set_model_initialization(keras.callbacks.Callback):
    def __init__(self, is_model_fit_var):
        self.is_model_fit_var = is_model_fit_var
    def on_train_begin(self, logs=None):
        self.is_model_fit_var = True
#########################Doc2Vec
class doc2vec:
    def generate_document_skipgrams(self, shuffle = False):
        #set our indecies to a unique key for the document
        num_docs = len(self.documents)
        sents = {self.vocabulary_size-num_docs+key:val for (key, val) in enumerate(self.documents)}#place these after the vocab size
        tokenizer = Tokenizer(num_words=self.vocabulary_size-num_docs, lower=True, filters=self.filters)
        tokenizer.fit_on_texts(sents.values())
        self.word_index_inv = {v: k for k, v in tokenizer.word_index.items()}
        
        #make names for the documents enumerated
        for i in range(len(self.documents)):
            self.word_index_inv[i+self.vocabulary_size-num_docs] = str('document_{}'.format(i))
            
        
        sequences = {key:tokenizer.texts_to_sequences([val])[0] for (key, val) in sents.items()}
        couples = []
        labels = []
        for doc, seq in sequences.items():
            for word in seq:
                couples.append((doc,word))
                labels.append(1)#this word is in the sentence
                for i in range(self.neg_samples):#try it many times depending on the size of dataset
                    ran_tok = random.randint(1, self.vocabulary_size-num_docs)
                    while(ran_tok in seq):#if it is in seq check another one
                        ran_tok = random.randint(1, self.vocabulary_size-num_docs)
                    couples.append((doc, ran_tok))
                    labels.append(0)
        doc_grams = pd.DataFrame({'docID': [i[0] for i in couples], 'word': [i[1] for i in couples], 'label': labels})
        if shuffle:
            doc_grams.sample(frac=1)
        print(len(doc_grams))
        word_target = doc_grams['docID']
        word_context = doc_grams['word']
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        labels = doc_grams['label'].tolist()
        return word_target, word_context, labels
    
    def fit_doc2vec(self, documents):
        self.documents = documents
        self.vocabulary_size += len(documents)+1
        word_target, word_context, labels = self.generate_document_skipgrams()
        #self.word_target = word_target
        #self.word_context = word_context
        #self.labels = labels
        self.model, self.validation_model = self.get_model()
        
        loss = self.model.fit([word_target, word_context], labels, batch_size=64, epochs=self.epochs)
                
        np.savetxt(str(self.embedding_file_location + '.csv'), self.model.layers[2].get_weights()[0], delimiter=",")
        self.model.save_weights(str(self.embedding_file_location + '.h5'))
        return self.model    

    def run_doc_sim(self, docID = None):
        doc_start_idx = self.vocabulary_size-len(self.documents)
        self.valid_docs = range(doc_start_idx, self.vocabulary_size)
        if docID == None:
            random_docs = np.random.choice(self.valid_docs, 5, replace=False)
        else:
            if type(docID) is not list:
                docID = [docID]
            
            random_docs = [doc_start_idx + _id for _id in docID]
            
        print (random_docs)
        top_k = 8
        for doc in random_docs:
            sim = self._get_sim_docs(doc)
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % self.word_index_inv[doc]
            for k in range(top_k):
                close_word = self.word_index_inv[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
    
    def _get_sim_docs(self, rand_docs):
        sim = np.zeros((self.vocabulary_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in self.valid_docs:
            in_arr1[0,] = rand_docs
            in_arr2[0,] = i
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

