import os
import inspect
from net import Net
from utils import Utils
import copy
import matplotlib.pyplot as plt
import time

utils = Utils()

IRIS_TRAIN_FILE = os.getcwd()+'/../data/iris-train.txt'
IRIS_TEST_FILE = os.getcwd()+'/../data/iris-test.txt'

# IDENTITY_TRAIN_FILE = os.getcwd()+'/identity-train.txt'
# IDENTITY_TEST_FILE

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def calculate_accuracy(inputs, targets):
    # predictions = []
    n_correct = 0
    n_samples = 0
    for inp, target in zip(inputs, targets):
        n_samples += 1
        pred = (net.feedForward(inp))
        
        for i in range(len(pred)):
            pred[i] = float("{:.1f}".format(pred[i]))

        true_max = target.index(max(target))
        pred_max = pred.index(max(pred))

        if true_max == pred_max:
            n_correct += 1
        # print('------------------')
    return n_correct/n_samples

def process_iris_output(outputs):
    for i in range(len(outputs)):
        if outputs[i] == 'Iris-setosa':
            outputs[i] = [1,0,0]
        elif outputs[i] == 'Iris-versicolor':
            outputs[i] = [0,1,0]
        elif outputs[i] == 'Iris-virginica':
            outputs[i] = [0,0,1]
    return outputs

def get_data(path):
    data = utils.load_examples(path)
    inputs = []
    outputs = []
    # utils.log('data', data)
    for example in data:
        inputs.append(example[:-1])
        outputs.append(example[-1])

    return utils.toFloat(inputs), process_iris_output(outputs)

if __name__ == '__main__':
    opt = utils.arg_parse() # get hyper-parameters
    hidden_arch = utils.get_hidden_arch(opt.hidden_arch) #forma user-defined hidden architecture
    inputs, outputs = get_data(IRIS_TRAIN_FILE)

    n_in = len(inputs[0])
    n_out = len(outputs[0])
    net_arch = [n_in] + hidden_arch + [n_out]
    utils.log('net_arch', net_arch)
    utils.log(f'Training with  {opt.algorithm}', None)
    net = Net(net_arch, lr=float(opt.lr), maxEpoch=int(opt.max_iter), momentum=float(opt.momentum), verbose=bool(opt.verbose), debug=opt.debug, algorithm=opt.algorithm, pr=float(opt.pr))
    
    # for w in net.w:
    #     print(w)
    
    # for a in (net.a):
    #     print(a)

    print('Training...')
    start = time.time()
    net.train(inputs, outputs)
    end = time.time()
    print(f'Trained in {(end - start)} (s)')

    # for w in net.w:
    #     print(w)
    #     print('-'*50)
    # print('reversed')
    # w_reversed = copy.deepcopy(net.w)
    # w_reversed.reverse()
    # # print(w_reversed)
    # for w in w_reversed:
    #     print(w)
    #     print('-'*50)
    
    # print('activations')
    # for a in net.a:
    #     print(a)
    #     print('-'*50)
    
    # print('Unit values')
    # for x in net.x:
    #     print(x)
    #     print('-'*50)

    plt.plot(net.lossHistory)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    # #calculate accuracy
    acc = calculate_accuracy(inputs, outputs)
    utils.log('Train Acc', acc)
    print('-'*50)
    inputs, outputs = get_data(IRIS_TEST_FILE)
    acc = calculate_accuracy(inputs, outputs)
    utils.log('Test Acc', acc)


