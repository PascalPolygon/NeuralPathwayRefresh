import inspect
import argparse

class Utils:
    def __init__(self):
        self.data = []
    
    def toFloat(self, data):
        for i, example in enumerate(data):
            data[i] = list(map(float, example))
        return data
        
    def log(self, name, data):
        TAG = inspect.stack()[1][3] #Name of function who called
        print(f'{TAG} {name} - {data}')

    def load_examples(self, file):
        trainingExamples = []
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                example = line.split(' ')
                trainingExamples.append(example)
        return trainingExamples
    
    def get_hidden_arch(self, hidden_arch):
        hidden_arch = hidden_arch.split('-') #Architecture of hidden layers (user defined)
        for i in range(len(hidden_arch)):
            hidden_arch[i] = int(hidden_arch[i])
        return hidden_arch

    def arg_parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--max_iter", help="max training iterations", default=500)
        parser.add_argument("--lr", help="learning rate", default=0.1)
        parser.add_argument("--momentum", help="momentum hyperparameter", default=0)
        parser.add_argument("--hidden_units", help="number of hidden units", default=3)
        parser.add_argument("--validation", help="percentage of data to keep for validation", default=0)
        parser.add_argument("--hidden_arch", help="customer hidden layers architecture. Format: #units-#units-#units (e.g. 1-3-2)", default='3')
        parser.add_argument("--verbose", help="verbose", default=False)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--loss_thresh", help="Maximum difference between loss of bestweights vs training weights on validation data", default=0.1)
        parser.add_argument("--algorithm", help="Training algorithm to use: One of [bp, npr]", default="bp")
        parser.add_argument("--pr", help="Probability with which strongest weight will be selected", default=0.8)
        
        opt = parser.parse_args()
        if opt.verbose == 'False': 
            opt.verbose = False
        elif opt.verbose =='True':
            opt.verbose = True

        return opt