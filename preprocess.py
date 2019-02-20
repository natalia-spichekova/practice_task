import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

patterns_to_replace = [
        (r'he\'s', 'he is'),
        (r'she\'s', 'she is'),
        (r'it\'s', 'it is'),
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', "i am"),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        #(r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)n\'re', '\g<1> are'),
        (r'(\w+)n\'d', '\g<1> would'),
        
        (r'<br \/>', ' '),
        (r'[()/{}\[\]\|@,:;?!-.]', ' '),
        (r'[^0-9a-z #+_]', ''),
        (r' +', ' ')
        ]

class TextPreprocessor():
    
    def __init__(self, patterns=patterns_to_replace):
        self.patterns=[(re.compile(regexpr), ex) for (regexpr, ex) in patterns]
    
    def reg_replace(self, text):
        s = text.lower()
        for (pattern, ex) in self.patterns:
            s = re.sub(pattern, ex, s)
        return s
    
    def tokenize(self, text):
        STOPWORDS = set(stopwords.words('english'))
        text = " ".join([word for word in word_tokenize(text) if word not in STOPWORDS])
        return text
    
    def tokenize_prepr(self, text):
        text = self.reg_replace(text)
        text = self.tokenize(text)
        return text
    
        