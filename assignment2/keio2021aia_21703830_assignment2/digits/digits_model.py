from . import *


class Digits_Model:

    def __init__(self, dim_input=784, dim_hidden=[100], dim_out=10):

        self.layers = []

        self.layers.append(Linear(dim_input, dim_hidden[0]))
        self.layers.append(Tanh())

        for i in range(0, len(dim_hidden)-1):
            self.layers.append(Linear(dim_hidden[i], dim_hidden[i+1]))
            self.layers.append(Tanh())

        self.layers.append(Linear(dim_hidden[len(dim_hidden)-1], dim_out))

        self._predict = softmax

    def __str__(self):

        return "Simple Neural Network\n\
        \tInput dimension: %d\n\
        \tHidden dimensions: %d\n\
        \tOutput dimension: %d\n" % (self._dim_input, self._dim_hidden, self._dim_out)

    def __call__(self, x):

        prediction = None

        outputs = self.forward(x)

        prediction = self._predict(outputs)
        pred = []

        for i in prediction:
            pred.append(np.argmax(i))
        prediction = np.array(pred)

        return prediction

    def load_model(self, file_path):

        with open(file_path, mode='rb') as f:
            loaded_model = pkl.load(f)

        self.__dict__.update(loaded_model.__dict__)

    def save_model(self):

        for layer in self.layers:
            layer.inputs = None
##
        with open('results/digits_model.pkl', 'wb') as f:
            pkl.dump(self, f)

    def forward(self, inputs):

        outputs = None
        outputs = inputs

        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs

    def backward(self, grad):

        for layer in reversed(self.layers):

            grad = layer.backward(grad)

        return grad

    def params_and_grads(self):

        for layer in self.layers:

            for name, param in layer.params.items():

                grad = layer.grads[name]
                yield param, grad
