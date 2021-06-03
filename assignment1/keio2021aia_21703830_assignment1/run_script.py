##
from sonar.sonar_data import *
from sonar.sonar_model import *
from sonar.sonar_trainer import *
from cats.cat_data import *
from cats.cat_model import *
from cats.cat_trainer import *
##

##


def main():

    Data = Sonar_Data()

    model = Sonar_Model(dimension=len(
        [next(iter(Data))][0][0][0]), activation=perceptron)

    trainer = Sonar_Trainer(next(iter(Data)), model)

    costs, accuracies = trainer.train(0.1, 500)

    model.save_model()

    Data = Cat_Data()
    model = Cat_Model(dimension=len(
        [next(iter(Data))][0][0][0]), activation=sigmoid)
    trainer = Cat_Trainer(Data, model)
    costs, accuracies = trainer.train(0.1, 80)
    model.save_model()
##


##
if __name__ == "__main__":
    main()
