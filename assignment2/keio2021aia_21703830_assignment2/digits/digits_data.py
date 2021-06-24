from . import *


class Digits_Data:

    def __init__(self, relative_path='../data/', data_file_name='digits_data.pkl', batch_size=64):

        self.batch_size = batch_size
        self.index = -1

        with open('%s%s' % (relative_path, data_file_name), mode='rb') as f:

            digits_data = pkl.load(f)

        self.samples = []

        train = digits_data['train']
        for j in range(10):
            for i in train[j]:

                c = np.zeros(10)

                c[j] = 1
                self.samples.append([i.flatten(), c])

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
        batch = []
        tar = []

        for index in self.starts:
            batch.append(self.samples[index][0])
            tar.append(self.samples[index][1])

        inputs = batch
        targets = tar

        return {'inputs': inputs, 'targets': targets}

    def shuffle(self):

        random.shuffle(self.samples)
