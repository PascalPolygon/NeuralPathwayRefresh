import os
import inspect
from net import Net
from utils import Utils

utils = Utils()

TENNIS_TRAIN_FILE = os.getcwd()+'/../data/tennis-train.txt'
TENNIS_TEST_FILE = os.getcwd()+'/../data/tennis-test.txt'


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def calculate_accuracy(inputs, outputs):
    n_correct = 0
    n_samples = len(inputs)
    for inp, target in zip(inputs, outputs):
        pred = net.feedForward(inp)
        pred[0] = round(pred[0])
        if target[0] == pred[0]:
            n_correct += 1
    return n_correct/n_samples

def process_tennis_data(inputs, outputs):
    for i, input in enumerate(inputs):
        new_input = []
        #Outlook
        if input[0] == 'Sunny':
            # inputs[i][0] = [1,0,0]
            new_input.append(1)
            new_input.append(0)
            new_input.append(0)
        elif input[0] == 'Overcast':
            # inputs[i][0] = [0,1,0]
            new_input.append(0)
            new_input.append(1)
            new_input.append(0)
        elif input[0] == 'Rain':
            # inputs[i][0] = [0,0,1]
            new_input.append(0)
            new_input.append(0)
            new_input.append(1)

        #Temperature
        if input[1] == 'Hot':
            # inputs[i][1] = [1,0,0]
            new_input.append(1)
            new_input.append(0)
            new_input.append(0)
        elif input[1] == 'Mild':
            # inputs[i][1] = [0,1,0]
            new_input.append(0)
            new_input.append(1)
            new_input.append(0)
        elif input[1] == 'Cool':
            # inputs[i][1] = [0,0,1]
            new_input.append(0)
            new_input.append(0)
            new_input.append(1)
        
        #Humidity
        if input[2] == 'High':
            # inputs[i][2] = [1,0]
            new_input.append(1)
            new_input.append(0)
        elif input[2] == 'Normal':
            inputs[i][2] = [0,1]
            new_input.append(0)
            new_input.append(1)
        
        #Wind
        if input[3] == 'Weak':
            # inputs[i][3] = [1,0]
            new_input.append(1)
            new_input.append(0)
        elif input[3] == 'Strong':
            # inputs[i][3] = [0,1]
            new_input.append(0)
            new_input.append(1)

        inputs[i] = new_input
        
    for i in range(len(outputs)):
        outputs[i] = [0] if outputs[i] == 'No' else [1]
    
    return inputs, outputs

def get_data(path):
    data = utils.load_examples(path)
    inputs = []
    outputs = []
    # utils.log('data', data)
    for example in data:
        inputs.append(example[:-1])
        outputs.append(example[-1])

    return process_tennis_data(inputs, outputs)

if __name__ == '__main__':
    opt = utils.arg_parse() # get hyper-parameters
    hidden_arch = utils.get_hidden_arch(opt.hidden_arch) #forma user-defined hidden architecture
    
    inputs, outputs = get_data(TENNIS_TRAIN_FILE)

    n_in = len(inputs[0])
    n_out = len(outputs[0])
    net_arch = [n_in] + hidden_arch + [n_out]
    utils.log('net_arch', net_arch)
    net = Net(net_arch, lr=float(opt.lr), maxEpoch=int(opt.max_iter), momentum=float(opt.momentum), verbose=bool(opt.verbose))
    print('Training...')
    net.train(inputs, outputs)

    # plt.plot(net.lossHistory)
    # plt.show()

    #calculate accuracy
    acc = calculate_accuracy(inputs, outputs)
    utils.log('Train Acc', acc)
    print('-'*50)
    inputs, outputs = get_data(TENNIS_TEST_FILE)
    acc = calculate_accuracy(inputs, outputs)
    utils.log('Test Acc', acc)
