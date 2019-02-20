from sklearn.datasets import load_files

def loaddata(foldername):
    """
        foldername: a folder with files
        
        return: tuple with two items:
                    list of reviews,
                    numpy.ndarray with sentiment assessment
    """
    reviews_data = load_files(foldername, categories=["neg", "pos"], encoding='utf-8', shuffle=False)
    return reviews_data.data, reviews_data.target
    
    
    
    
    
    