from . import *

class Sonar_Data:

    def __init__(self, data_file_path='', data_file_name='sonar_data.pkl'):
                
        self.data = []

        YOUR_CODE

        self.shuffle()

    def __iter__(self):
        
        return self

    def __next__(self):
        
        YOUR_CODE

    def shuffle(self):
        
        random.shuffle(self.data)
    
    def __len__(self):
        
        return len(self.data)
