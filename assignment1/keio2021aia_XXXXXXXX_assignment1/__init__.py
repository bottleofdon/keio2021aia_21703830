import numpy as np
import pickle as pkl
import random


def perceptron(z):
    return -1 if z <= 0 else 1


def ploss(yhat, y):
    return max(0, -yhat*y)


def ppredict(self, x):
    return self(x)


class Sonar_Model:

    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x), predict=ppredict):

        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)*np.sqrt(1/dimension)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):

        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)

    def __call__(self, x):

        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)

        return yhat

    def load_model(self, file_path):

        with open(file_path, mode='rb') as f:
            mm = pkl.load(f)
        self._dim = mm._dim
        self.w = mm.w
        self.b = mm.b
        self._a = mm._a

    def save_model(self):
        f = open('sonar_model.pkl', 'wb')
        pkl.dump(self, f)
        f.close


class Sonar_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = ploss

    def accuracy(self, data):

        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])

    def train(self, lr, ne):

        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))

        costs = []
        accuracies = []

        for epoch in range(ne):
            J = 0
            for x, y in self.dataset:

                x = np.array(x)
                yhat = self.model(x)
                J += self.loss(self.model.predict(x), y)
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)

            J /= len(self.dataset)

            accuracy = self.accuracy(self.dataset)
            if epoch % 10 == 0:
                print('--> epoch=%d, accuracy=%.3f' % (epoch, accuracy))
            costs.append(J)
            accuracies.append(accuracy)

        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))

        costs = list(map(lambda t: np.mean(t), [np.array(costs)[
                     i-10:i+11] for i in range(1, len(costs)-10)]))
        accuracies = list(map(lambda t: np.mean(t), [np.array(accuracies)[
                          i-10:i+11] for i in range(1, len(accuracies)-10)]))

        return (costs, accuracies)


class Sonar_Data:

    def __init__(self, data_file_path='', data_file_name='sonar_data.pkl'):

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
