from . import *

class Digits_Data:

    def __init__(self, relative_path='../data/', data_file_name='digits_data.pkl', batch_size=64):

        self.batch_size = batch_size
        self.index = -1
        
        with open('%s%s' % (relative_path, data_file_name), mode='rb') as f:
            
            digits_data = pkl.load(f)
            
        self.samples = []
        
        YOUR_CODE # hint: you will need to flatten the images to represent them as vectors (numpy arrays) and pair them with digit labels from the training data
                
        self.shuffle()
        
        self.starts = np.arange(0, len(self.samples), self.batch_size)

    def __iter__(self):
                
        return self

    def __next__(self):

        self.index += 1
        
        if self.index + 1 > len(self.starts):
            
            self.index = -1
            self.shuffle()
            raise StopIteration
            
        inputs = None
        targets = None
        YOUR_CODE # hint: use the starts initialized in the last line of the constructor and the batch size to generate a batch of inputs and the corresponding batch of targets

        return {'inputs': inputs, 'targets': targets}

    def shuffle(self):
        
        random.shuffle(self.samples)
