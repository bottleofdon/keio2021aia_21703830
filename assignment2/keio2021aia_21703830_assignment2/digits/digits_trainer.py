from . import *


class Digits_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = CrossEntropy()

    def accuracy(self):

        acc = None

        dat = []
        for data in self.dataset:
            dat.append([data['inputs'], data['targets']])

        acc = 0

        for x, y in dat:
            pred = self.model(x)
            acc += 100*float(sum([1 for i in range(len(pred)) if pred[i]
                             == np.argmax(y[i], axis=0)])/float(len(pred)))

        acc = acc/len(dat)

        return acc

    def step(self, lr):

        for param, grad in self.model.params_and_grads():

            param -= lr * grad

    def train(self, lr, ne):

        print('initial accuracy: %.3f\n\n' % (self.accuracy()))

        print('training model on data...\n')
        print('='*80+'\n')

        for epoch in range(1, ne + 1):

            epoch_loss = 0.0

            for batch in self.dataset:
                predicted = None

                predicted = self.model.forward(np.array(batch['inputs']))

                epoch_loss += self.loss.loss(predicted,
                                             np.array(batch['targets']))

                grad = None

                grad = self.loss.grad(predicted, np.array(batch['targets']))

                self.model.backward(grad)
                self.step(lr)

            print("""epoch %d:\n
            \t loss = %.3f\n
            \t accuracy=%.3f""" % (epoch, epoch_loss, self.accuracy()))

        print('='*80+'\n')
        print('training complete!\n\n')
        print('final accuracy: %.3f' % (self.accuracy()))
