##
from digits.digits_data import * 
from digits.digits_model import * 
from digits.digits_trainer import * 
##

##
def main():

    data = Digits_Data()
    model = Digits_Model(YOUR_CODE) # hint: describe the shape of your simple neural network by a list of hidden layer sizes
    trainer = Digits_Trainer(data, model)
    trainer.train(YOUR_CODE) # hint: choose a learning rate and the number of epochs to train for
    model.save_model()
##

##
if __name__ == "__main__":
    main()
##
