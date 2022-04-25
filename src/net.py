from cmath import nan
import random
from math import e
from utils import Utils
import copy

utils = Utils()
class Net():
    def __init__(self, n_neurons, lr=0.01, maxEpoch = 100, momentum = 0, verbose=True, debug=False, algorithm='bp'):
        self.n_neurons = n_neurons
        self.verbose = verbose
        self.eta = lr
        self.maxEpoch = maxEpoch
        self.alpha = momentum #Momentum hyper parametere
        self.debug = debug
        self.algorithm = algorithm

        self.w = [] #weights
        self.a = [] #activations (value at unit only for hidden and output)
        self.x = [] #Unit values (Includes input and biases)

        self.lossHistory = []

        for i in range(len(n_neurons)-1):
            self.w.append(self.initLayerWeights(n_neurons[i], n_neurons[i+1]))
        
        # self.weight_update = copy.copy(self.w)

    def initLayerWeights(self, n_in_units, n_out_units):
        layerWeights = []
        for j in range(n_out_units):
            weights = []
            for i in range(n_in_units+1):
                weights.append(random.uniform(-0.5, 0.5))
            layerWeights.append(weights)
        return layerWeights
    
    def sigmoid(self, x):
        return (1/(1+e**-x))
    
    def relu(self, x):
        return max(0.0, x)

    def reluDerivative(self, x):
        if x < 0:
            return 0.0
        elif x > 0:
            return 1.0
        elif x == 0.0:
            return 0.0
        # elif x == 0.0:
        #     return float("nan")

    def loss(self, outputs, targets):
        E = 0
        for output, target in zip(outputs, targets):
            for o_k, t_k in zip(output, target):
                E += (t_k - o_k)**2
        return E/2
    
    def train(self, inputs, targets, validationSet=None, lossThresh=5):

        self.lossHistory = []

        for epoch in range(self.maxEpoch):
            
            w_update = [] #Weight update at current iter. Used for adding momentum 

            outputs = []
            # outputs = self.tune_weights(inputs, targets)
            for i, (input, target) in enumerate(zip(inputs, targets)):
                out = self.feedForward(input)
                # print('x')
                # print(self.x)
                if self.debug:
                    print(out)
                outputs.append(out)
                if self.algorithm == 'bp':
                    w_update.append(self.backPropagate(target))
                elif self.algorithm == 'npr':
                    w_update.append(self.refreshNeuralPathways(out, target))
                # if epoch == 0:
                #     # w_update = self.backPropagate(target, i, 0) #No delta_w(n-1) on first iteration
                #     w_update.append(self.backPropagate(target)) #No delta_w(n-1) on first iteration
                # else:
                #     # w_update = self.backPropagate(target, i, prev_w_update)
                #     w_update.append(self.backPropagate(target, prev_w_update[i])) #previous weight update of this sample
                    
            # prev_w_update = copy.deepcopy(w_update)

            loss = self.loss(outputs, targets)
            self.lossHistory.append(loss)

            #Keep 2 copies of weights: training, and best performing weights
            #Once training weights reach significantly higher error over stored weights -> termninate
            if validationSet is not None:
                valInputs = validationSet[0]
                valTargets = validationSet[1]

                valOutputs = []
                for input in valInputs:
                    valOutput = self.feedForward(input)
                    valOutputs.append(valOutput)

                valLoss = self.loss(valOutputs, valTargets)
                if epoch == 0:
                    bestValLoss = valLoss
                    bestWeights = copy.copy(self.w)
                else:
                    if (valLoss - bestValLoss) > lossThresh: #valLoss is greater than bestLoss
                        #Return stored weights and terminate
                        self.w = copy.copy(bestWeights)
                        utils.log(f'** Terminating training, best weights found at epoch {epoch}', None)
                        break
                    elif valLoss < bestValLoss:
                        bestValLoss = valLoss
                        bestWeights = copy.copy(self.w)

            if self.verbose:
                utils.log('epoch', epoch)
                utils.log('loss', loss)
                print('-'*10)


    def feedForward(self, inps):
        self.x = []
        self.a = []
        inputs = copy.copy(inps)
        if len(inputs) != self.n_neurons[0]:
            raise ValueError(f'[ERROR] Input vector of size {len(inputs)} mismatch network input of size {self.n_neurons[0]}')
        layerActivations = []

        prevLayerActivations = inputs #Previous layer activations
        layer_i = 0
        for layerWeights in self.w:
            layerActivations = []
            for i, unitInWeights in enumerate(layerWeights):
                unitOut = 0
                #Summation loop
                for input, w in zip(prevLayerActivations, unitInWeights[:-1]): 
                    unitOut += input*w
                unitOut += unitInWeights[-1] #bias
                layerActivations.append(self.sigmoid(unitOut))
                # if layer_i == len(layerWeights)-1: #This is the last layer, use sigmoid
                #     layerActivations.append(self.sigmoid(unitOut))
                # else:
                #     layerActivations.append(self.relu(unitOut))
            self.a.append(layerActivations)
            prevLayerActivations = layerActivations
            layer_i +=1
        
        #Create unit value list
        inputs.append(1) #Bias unit
        self.x.append(inputs)
        for i, a in enumerate(self.a):
            if (i < len(self.a)-1): #Not Output units
                a.append(1) #bias unit
            self.x.append(a)
        
        return self.a[-1] #Only return last layer (outut layer) activation
    
    def backPropagate(self, target, prev_weight_update=[]):
        # utils.log('Training w bp', None)
        delta_out = []
        
        out = self.x[-1] #Last units are output
        for o, t in zip(out, target):
            delta_out.append(o*(1-o)*(t-o)) #Derivative of sigmoid (always use sigmoid for output layer)
            # delta_out.append(self.reluDerivative(o)*(t-o)) #Derivative of ReLU

        net_out = delta_out
        #Find errors in hiddden units
        hiddenUnits = self.x[1:-1] #Exclude input and output layers
        
        n_layers = len(hiddenUnits) #Number of hidden layers
        deltas = []
        for layer in range(n_layers, 0, -1):
            units = hiddenUnits[layer-1]
            delta_layer = []
            for h, o_h in enumerate(units):
                sensitivity = 0
                # utils.log('delta_out', delta_out)
                for k, delta in enumerate(delta_out): #Should be prev/next layer not delta_out
                    sensitivity += self.w[layer][k][h]*delta
                delta_layer.append(o_h*(1-o_h)*sensitivity)
                # delta_layer.append(self.reluDerivative(o_h)*sensitivity)
            deltas.append(delta_layer)
            delta_out = delta_layer[:1]

        deltas.reverse() # reverse because you calculated deltas (out -> in), but weights are updated (in->out)
        deltas.append(net_out)

        weight_update = copy.deepcopy(self.w)

        for j in range(len(self.w)): #Layer
            for k in range(len(self.w[j])): #Unit
                for i in range(len(self.w[j][k])): #Weight
                    # Calculate weight update
                    weight_update[j][k][i] = self.eta*deltas[j][k]*self.x[j][i]
                    #Update weights
                    if prev_weight_update:
                        self.w[j][k][i] += weight_update[j][k][i] + self.alpha*prev_weight_update[j][k][i] #Gradient descent w momentum
                    else:
                        self.w[j][k][i] += weight_update[j][k][i]
        # if self.verbose:
        #     print(f'New weights {self.w[0]}')
    
    #Simple: weight only
    # def refreshNeuralPathways(self, output, target):
    #     for i, x in enumerate(zip(output, target)):
    #         o = x[0]
    #         t = x[1]
    #         err = t-o #Use different loss func if needed
    #         #update the entire pathway w err
    #         #Find next weight to use
    #         #reverse weights because we want to start w weights at ouput layer and work our way backwards (similar to backProp)
    #         w_reversed = copy.deepcopy(self.w)
    #         w_reversed.reverse()

    #         a_reversed = copy.deepcopy(self.a)
    #         a_reversed.reverse()
    #         # utils.log('a_reversed', a_reversed)
    #         curr_hop_id = i
    #         # print(i)

    #         """
    #             unit_input_activations = a_reversed[layer] #From this unit's persepective looking at activations from the prev layers that feed into it
    #             a_reversed includes output activations, should we not skip the very first one (last one before you reverse)
    #         """
    #         for layer, _ in enumerate(w_reversed):
    #             # print(layer)
    #             unit_weights = w_reversed[layer][curr_hop_id] 
    #             # utils.log('unit_input_activations', unit_input_activations)
    #             #Get the strongest weight (Weight most responsible for the err)
    #             strongest_weight = max(unit_weights[:-1]) #Ignore last weight because it is bias (TODO: thinks about how else you want to handle bias weights - maybe just ignoring them isn't the best for best performance)
    #             next_hop_id = unit_weights.index(strongest_weight) 
    #             # unit_weights[next_hop_id] += err #update strongest  
    #             # w_reversed[layer][curr_hop_id][next_hop_id] += self.eta*err*unit_input_activations[next_hop_id] #Try w and w/o unit_input_activation (xji  in T4.5 from the book)
    #             w_reversed[layer][curr_hop_id][next_hop_id] += self.eta*err #Try w and w/o unit_input_activation (xji  in T4.5 from the book)
    #             curr_hop_id = next_hop_id
        
    #     w_reversed.reverse() #Reverse back to feedForward order
    #     self.w = copy.deepcopy(w_reversed)

#Simpl: weights only
    # def refreshNeuralPathways(self, output, target):
    #     # utils.log('Training w rpr', None )
    #     for i, x in enumerate(zip(output, target)):
    #         o = x[0]
    #         t = x[1]
    #         err = t-o #Use different loss func if needed
    #         delta_out = o*(1-o)*(t-o)
    #         """
    #             unit_input_activations = a_reversed[layer] #From this unit's persepective looking at activations from the prev layers that feed into it
    #             a_reversed includes output activations, should we not skip the very first one (last one before you reverse)
    #         """
    #         # for i in range( len(wordList) - 1, -1, -1) :
    #         #     print(wordList[i])
    #         hiddenUnits = self.x[1:-1] #Exclude input and output layers
    #         curr_conn_id = i
    #         for layer in range(len(self.w)-1, -1, -1): #Loop backwards
    #             # print(layer)
    #             unit_weights = self.w[layer][curr_conn_id] 
    #             # utils.log('unit_input_activations', unit_input_activations)
    #             #Get the strongest weight (Weight most responsible for the err)
    #             strongest_weight = max(unit_weights[:-1]) #Ignore last weight because it is bias (TODO: thinks about how else you want to handle bias weights - maybe just ignoring them isn't the best for best performance)
    #             next_conn_id = unit_weights.index(strongest_weight) 

    #             sensitivity = strongest_weight*delta_out
    #             o_h = hiddenUnits[next_conn_id]
    #             delta = o_h*(1-o_h)*sensitivity #GD
    #             # unit_weights[next_hop_id] += err #update strongest  
    #             # w_reversed[layer][curr_hop_id][next_hop_id] += self.eta*err*unit_input_activations[next_hop_id] #Try w and w/o unit_input_activation (xji  in T4.5 from the book)
    #             self.w[layer][curr_conn_id][next_conn_id] += self.eta*delta*hiddenUnits[next_conn_id]
    #             # self.w[layer][curr_hop_id][next_hop_id] += self.eta*err #Try w and w/o unit_input_activation (xji  in T4.5 from the book)
    #             curr_conn_id = next_conn_id

    def refreshNeuralPathways(self, output, target):
        # utils.log('Training w rpr', None )
        updates_ref = []
        updates = []
        units = self.x[1:] #Exclude input layer of unit output values

        for i, x in enumerate(zip(output, target)):
            o = x[0]
            t = x[1]
            
            delta = o*(1-o)*(t-o)

            curr_conn_id = i
            for layer in range(len(self.w)-1, -1, -1): #Loop backwards
                unit_weights = self.w[layer][curr_conn_id] 
                strongest_weight = max(unit_weights[:-1]) #Ignore last weight because it is bias (TODO: thinks about how else you want to handle bias weights - maybe just ignoring them isn't the best for best performance)
                next_conn_id = unit_weights.index(strongest_weight) 

                sensitivity = strongest_weight*delta #Sensitivity of strongest weight
                o_h = units[layer][next_conn_id] #Hidden unit that stronges weight leads to
                delta = o_h*(1-o_h)*sensitivity #GD
                updates.append(self.eta*delta*o_h) #Store the update
                updates_ref.append([layer, curr_conn_id, next_conn_id]) #Store references of weights to udpates
                curr_conn_id = next_conn_id

        #Apply updates (only to weights of ids)
        for update, update_ref in zip(updates, updates_ref):
            self.w[update_ref[0]][update_ref[1]][update_ref[2]] += update




