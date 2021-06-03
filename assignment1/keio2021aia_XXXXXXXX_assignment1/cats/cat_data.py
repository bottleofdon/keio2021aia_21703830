from . import *


class Cat_Data:

    def __init__(self, data_file_path='', data_file_name='cat_data.pkl'):

        self.data = []
        self.train_mean = 0.0
        self.train_sd = 0.0

        filepath = data_file_path+data_file_name

        with open(filepath, 'rb') as f:
            dat = pkl.load(f)
        dat = dat["train"]
        k = ["no_cat", "cat"]
        for i in k:
            for j in dat[i]:
                if i == "no_cat":
                    j = self.standardize(j)
                    s = j.flatten().tolist()
                    self.data.append([s, 0])
                else:
                    j = self.standardize(j)
                    s = j.flatten().tolist()
                    self.data.append([s, 1])

        self.shuffle()

    def __iter__(self):

        return self

    def __next__(self):

        return self.data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, i):

        return self.data[i]

    def shuffle(self):

        random.shuffle(self.data)

    def standardize(self, rgb_images):

        mean = np.mean(rgb_images, axis=(0, 1, 2), keepdims=True)
        sd = np.std(rgb_images, axis=(0, 1, 2), keepdims=True)
        return (rgb_images - mean) / sd
