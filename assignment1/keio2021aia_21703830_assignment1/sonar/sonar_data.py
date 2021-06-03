from . import *


class Sonar_Data:

    def __init__(self, data_file_path='../data/', data_file_name='sonar_data.pkl'):

        self.data = []

        filepath = data_file_path+data_file_name

        with open(filepath, 'rb') as f:
            dat = pkl.load(f)

        k = ["r", "m"]

        for i in k:
            for j in dat[i]:
                if i == "r":

                    self.data.append([j, 0])
                else:
                    self.data.append([j, 1])

        self.shuffle()

    def __iter__(self):
        return self

    def __next__(self):

        return self.data

    def shuffle(self):

        random.shuffle(self.data)

    def __len__(self):

        return len(self.data)
