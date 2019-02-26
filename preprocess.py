# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:23:45 2019

@author: natalia.spichekova

    Functions to clean text data.
"""

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# patterns to be replaced
patterns_to_replace = [
        (r'he\'s', 'he is'),
        (r'she\'s', 'she is'),
        (r'it\'s', 'it is'),
        (r'won\'t', 'will not'),
        (r'can\'t', 'can not'),
        (r'i\'m', "i am"),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would'),
        
        (r'<br \/>', ' '),
        (r'[()/{}\[\]\|@,:;?!-.]', ' '),
        (r'[^0-9a-z #+_]', '')
        ]

class TextPreprocessor():
    
    """
        Perform text cleaning: replace patterns in the string, remove stopwords.
    """
    def __init__(self, patterns=patterns_to_replace):
        """
            patterns_to_replace: list
        """
        
        self.patterns=[(re.compile(regexpr), ex) for (regexpr, ex) in patterns]
    
    def reg_replace(self, text):
        """
            Replace patterns in the string.
            
            text: string
            
            return: a modified string
        """
        
        s = text.lower()
        for (pattern, ex) in self.patterns:
            s = re.sub(pattern, ex, s)
        return s
    
    def remove_stopwords(self, text):
        """
            Remove stowords from the string.
            
            text: string
            
            return: a modified string
        """
        STOPWORDS = set(stopwords.words('english'))
        # exclude "no"-words from stopwords corpus
        for word in ["no", "nor", "not"]:
            STOPWORDS.remove(word)
        text = " ".join([word for word in word_tokenize(text) if word not in STOPWORDS])
        return text
    
    def clean_text(self, text):
        """
            Clean text: replace patterns, remove stopwords.
            
            text: string
            
            return: a modified string
        """
        text = self.reg_replace(text)
        text = self.remove_stopwords(text)
        return text
    
        