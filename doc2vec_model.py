# -*- coding: utf-8 -*-
"""
     Functions to deal with gensim.doc2vec model.
"""
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from random import shuffle

DOC2VEC_MODEL_PATH={"DOC2VEC": "doc2vec.model"}

class Doc2Vec_Model():
    """
        Wrapper to deal with densim.doc2vec model.
    """
        
    def doc2vec_train(self, x_train, max_epochs=10, dm=0, dm_concat=1, vec_size=100, window=5,\
                     negative=5, hs=0, min_count=2, sample=0, epochs=20, workers=1, alpha=0.025,\
                     min_alpha=0.025, model_file=DOC2VEC_MODEL_PATH["DOC2VEC"]):
        """
            Train and save Doc2Vec model.
            
            x_train: input samples
            max_epochs: number of epochs to train the model (it is number of explicit passes over the courpus 
                                                             with learning rate reduction, see https://markroxor.github.io/gensim/static/notebooks/doc2vec-IMDB.html)
            model_file: path to save the model
            All other parameters ar ethe same as in gensim.models.doc2vec.Doc2Vec.train function.
            
            return: trained model
            
        """
        model = Doc2Vec(dm=dm, dm_concat=dm_concat, vector_size=vec_size, window=window, \
                        negative=negative, hs=hs, min_count=min_count, sample=sample, \
                        epochs=epochs, workers=workers, alpha=alpha, min_alpha=min_alpha)
        # convert raw text into appropriate format
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_train)] 
        # build vocabulary
        model.build_vocab(tagged_data) 
        for epoch in range(max_epochs):
            # shuffling gets better results
            shuffle(tagged_data)
            if epoch % 2 == 0:
                print 'iteration {0}'.format(epoch)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
            # train the model as it advised at gensim blog post https://rare-technologies.com/doc2vec-tutorial/
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(model_file)
        print("Model Saved")
        
        return model
        
    def doc2vec_infer(self, model, x_test):
        """
            Infer vector representation for input documents.
            
            model: trained doc2vec model
            x_test: list of input documents
            
            return: list of vector representation for input documents.  
        """
        test_tok = [word_tokenize(item) for item in x_test]
        infered_vectors = [model.infer_vector(item) for item in test_tok]
        return infered_vectors
