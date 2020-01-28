import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        self.activation_function = sigmoid
    
    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
           
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs
            
            error = y - final_outputs
            output_error_term = error * 1.0
            hidden_error = np.dot(self.weights_hidden_to_output, error) 
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
            delta_weights_i_h += hidden_error_term * X[:,None]
            delta_weights_h_o += output_error_term * hidden_outputs[:,None]
            
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records
        
    def run(self, features):
        hidden_inputs =  np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs 
        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)

data = pd.read_csv('ohlc_1D.csv')
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)
data['Volume_(BTC)'].drop
quant_features = ['Open', 'High', 'Low', 'Close']
# Armazena as escalas em um dicionário para que possamos convertê-las novamente mais tarde
#Para facilitar o treinamento da rede, padronizaremos cada uma das variáveis contínuas. Ou seja, vamos mudar e dimensionar as variáveis de forma que elas tenham média zero e um desvio padrão de 1.

scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
print(data)

# Salve dados por aproximadamente os últimos 700 dias
test_data = data[-525:]

# Agora remova os dados de teste do conjunto de dados
data = data[:-525]
# Separe os dados em recursos e destinos
target_fields = ['Close']
features, targets = data.drop(target_fields, axis=1), data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
features.head()
# Segure os últimos 60 dias ou mais dos dados restantes como um conjunto de validação
train_features, train_targets = features[:-420], targets[:-420]
val_features, val_targets = features[-420:], targets[-420:]

###################treinamento
import sys

iterations = 100
learning_rate = 0.5
hidden_nodes = 10
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['Close']
                             
    network.train(X, y)
    
    train_loss = MSE(network.run(train_features).T, train_targets['Close'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['Close'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
    
plt.figure(figsize=(15,8))
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()

novo = test_features[-1:,]
plt.plot(test_data(['Close']),test_features)

mean, std = scaled_features['Close']

ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['Close']), label='Close')
ax.set_xlim(right=len(predictions))
ax.legend()
teste = MSE(network.run(test_features).T, test_targets['Close'].valupredictions = network.run(test_features).T*std + meanes)
print(teste)




#dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
#dates = dates.apply(lambda d: d.strftime('%b %d'))
#ax.set_xticks(np.arange(len(dates))[12::24])
#_ = ax.set_xticklabels(dates[12::24], rotation=45)