##
from digits.digits_data import *
from digits.digits_model import *
from digits.digits_trainer import *
##

##


def main():

    data = Digits_Data()
    # hint: describe the shape of your simple neural network by a list of hidden layer sizes
    model = Digits_Model(784, [400], 10)
    trainer = Digits_Trainer(data, model)
    # hint: choose a learning rate and the number of epochs to train for
    trainer.train(0.001, 15)
    model.save_model()
##


##
if __name__ == "__main__":
    main()
##
