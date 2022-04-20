import os
import inspect
from net import Net
from utils import Utils

utils = Utils()

TENNIS_TRAIN_FILE = os.getcwd()+'/../data/tennis-train.txt'
TENNIS_TEST_FILE = os.getcwd()+'/../data/tennis-test.txt'

IDENTITY_TRAIN_FILE = os.getcwd()+'/../data/identity-train.txt'
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

def disp_net_representation(inputs):
    print('              Input                      Hidden Values                        Output')
    for input in inputs:
        pred = net.feedForward(input)
        hiddenValues = net.a[-2][:-1] #Second to last of activations are hidden activations, exclude the very last one because it's the bias
        for i in range(len(pred)):
            pred[i] = float("{:.1f}".format(pred[i]))
        thresholded = []
        for i in range(len(hiddenValues)):
            hiddenValues[i] = float("{:.2f}".format(hiddenValues[i]))
            thresholded.append(1 if hiddenValues[i]  >= 0.5 else 0)
        # print(f'{*input} -> {hiddenValues} ({thresholded}) -> {pred}', sep=" ")
        print(*input, sep=" ", end=" ")
        print(' -> ',end=" ")
        print(*hiddenValues, sep=" ", end=" ")
        print('(', end="")
        print(*thresholded, sep=" ", end="")
        print(')', end=" ")
        print(' -> ',end=" ")
        print(*pred, sep=" ")
        
if __name__ == '__main__':
    data = utils.load_examples(IDENTITY_TRAIN_FILE)
    opt = utils.arg_parse() # get hyper-parameters
    hidden_arch = utils.get_hidden_arch(opt.hidden_arch) #forma user-defined hidden architecture

    inputs = []
    outputs = []
    for example in data:
        inputs.append(example[:8])
        outputs.append(example[9:])
    #Convert to float
    inputs = utils.toFloat(inputs)
    outputs = utils.toFloat(outputs)

    # utils.log('intput', inputs)
    # utils.log('output', outputs)

    n_in = len(inputs[0])
    n_out = len(outputs[0])
    # utils.log('net_arch', net_arch)
    net = Net([n_in, 3, n_out], lr=float(opt.lr), maxEpoch=int(opt.max_iter), verbose=bool(opt.verbose))
    print('Training...')
    net.train(inputs, outputs)

    # plt.plot(net.lossHistory)
    # plt.show()

    acc = calculate_accuracy(inputs, outputs)
    print('-'*50)
    utils.log('Train Acc (3 hidden units)', acc)
    print('-'*50)

    disp_net_representation(inputs)
    #Show weights
    # utils.log('Hidden Values', net.a)

    #Using 4 hidden units
    net = Net([n_in, 4, n_out], lr=5, maxEpoch=200, verbose=False)
    print('-'*50)
    utils.log('Training with 4 hidden units ...', None)
    net.train(inputs, outputs)
    acc = calculate_accuracy(inputs, outputs)
    print('-'*50)
    utils.log('Train Acc (4 hidden units)', acc)
    print('-'*50)
    disp_net_representation(inputs)
