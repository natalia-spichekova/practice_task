# -*- coding: utf-8 -*-
"""
    Auxiliary functions to deal with data.  
"""

from sklearn.datasets import load_files

def loaddata(foldername):
    """
        Load text files with categories from 'foldername' folder.
        
        foldername: string
        
        return: tuple with two items:
                    raw text data
                    numpy.ndarray with classification labels
    """
    reviews_data = load_files(foldername, categories=["neg", "pos"], encoding='utf-8', shuffle=False)
    return reviews_data.data, reviews_data.target
    

    
    
    
    
    