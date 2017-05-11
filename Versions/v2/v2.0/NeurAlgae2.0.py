# FlowTech | NeurAlgae
## 2017 CWSF Science Fair | NeurAlgae: HAB Prediction Using Machine Learning Algorithms

#Describes and trains a neural network for the analysis and prediction of algal bloom data
#Copyright (C) 2017 Zachary Trefler and Atif Mahmud

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

#If you have comments, questions, concerns, or you just want to say 'hi',
#email Zachary Trefler at zmct99@gmail.com or Atif Mahmud at atifmahmud101@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, regularizers
from keras.models import Model
#from Data import dataX as X, dataPN as PN, dataCDA as CDA, dataPDA as PDA
import Data

X = Data.dataX[5000:len(Data.dataX) - 5000]
PN = Data.dataPN[5000:len(Data.dataPN) - 5000]
CDA = Data.dataCDA[5000:len(Data.dataCDA) - 5000]
PDA = Data.dataPDA[5000:len(Data.dataPDA) - 5000]

Xr = np.zeroes((3, 2))
PNr = np.array

architecture = int(input("Which network architecture to use? "))

if architecture == 0:
    #Overfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu")(inputs)
    layer2 = Dense(64, activation = "relu")(layer1)
    outputs = Dense(1, activation = "sigmoid")(layer2)
    epochnum = 256
    minimizer = "rmsprop"
    cost = "mean_squared_error"
elif architecture == 1:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l1_l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l1_l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 256
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 2:
    #Overfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu")(inputs)
    layer2 = Dense(64, activation = "relu")(layer1)
    outputs = Dense(1, activation = "sigmoid")(layer2)
    epochnum = 256
    minimizer = "rmsprop"
    cost = "mean_squared_error"
elif architecture == 3:
    #Pretty good
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 64
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 4:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.001))(inputs)
    drop1 = Dropout(0.5)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.001))(drop1)
    drop2 = Dropout(0.5)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 5:
    #Surprisingly good underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu")(inputs)
    outputs = Dense(1, activation = "sigmoid")(layer1)
    epochnum = 1
    minimizer = "rmsprop"
    cost = "mean_squared_error"
elif architecture == 6:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l1(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l1(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 64
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 7:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0005))(inputs)
    drop1 = Dropout(0.33)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0005))(drop1)
    drop2 = Dropout(0.33)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 8:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.20)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.20)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 9:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    layer3 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop2)
    drop3 = Dropout(0.25)(layer3)
    outputs = Dense(1, activation = "sigmoid")(drop3)
    epochnum = 64
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 10:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    layer3 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop2)
    drop3 = Dropout(0.25)(layer3)
    outputs = Dense(1, activation = "sigmoid")(drop3)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 11:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 64
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 12:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 13:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.25)(layer1)
    layer2 = Dense(64, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.25)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 32
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 14:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.5)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.5)(layer2)
    outputs = Dense(1, activation = "sigmoid")(drop2)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 15:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.5)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.5)(layer2)
    layer3 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop2)
    drop3 = Dropout(0.5)(layer1)
    layer4 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop3)
    drop4  = Dropout(0.5)(layer1)
    outputs = Dense(1, activation = "sigmoid")(drop4)
    epochnum = 256
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 16:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(inputs)
    drop1 = Dropout(0.5)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop1)
    drop2 = Dropout(0.5)(layer2)
    layer3 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop2)
    drop3 = Dropout(0.5)(layer3)
    layer4 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop3)
    drop4 = Dropout(0.5)(layer4)
    layer5 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop4)
    drop5 = Dropout(0.5)(layer1)
    layer6 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop5)
    drop6 = Dropout(0.5)(layer2)
    layer7 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop6)
    drop7 = Dropout(0.5)(layer3)
    layer8 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001))(drop7)
    drop8 = Dropout(0.5)(layer4)
    outputs = Dense(1, activation = "sigmoid")(drop8)
    epochnum = 128
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 17:
    #Overfit
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(inputs)
    drop1 = Dropout(0.05)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop1)
    drop2 = Dropout(0.05)(layer2)
    layer3 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop2)
    drop3 = Dropout(0.05)(layer3)
    layer4 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop3)
    drop4 = Dropout(0.05)(layer4)
    layer5 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop4)
    drop5 = Dropout(0.05)(layer1)
    layer6 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop5)
    drop6 = Dropout(0.05)(layer2)
    layer7 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop6)
    drop7 = Dropout(0.05)(layer3)
    layer8 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00001))(drop7)
    drop8 = Dropout(0.05)(layer4)
    outputs = Dense(1, activation = "sigmoid")(drop8)
    epochnum = 64
    minimizer = "nadam"
    cost = "mean_squared_error"
elif architecture == 18:
    #Interesting
    inputs = Input(shape = (9,))
    layer1 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(inputs)
    drop1 = Dropout(0.2)(layer1)
    layer2 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop1)
    drop2 = Dropout(0.2)(layer2)
    layer3 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop2)
    drop3 = Dropout(0.2)(layer3)
    layer4 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop3)
    drop4 = Dropout(0.2)(layer4)
    layer5 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop4)
    drop5 = Dropout(0.2)(layer1)
    layer6 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop5)
    drop6 = Dropout(0.2)(layer2)
    layer7 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop6)
    drop7 = Dropout(0.2)(layer3)
    layer8 = Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.00005))(drop7)
    drop8 = Dropout(0.2)(layer4)
    outputs = Dense(1, activation = "sigmoid")(drop8)
    epochnum = 64
    minimizer = "nadam"
    cost = "mean_squared_error"
else:
    #Underfit
    inputs = Input(shape = (9,))
    layer1 = Dense(16, activation = "sigmoid")(inputs)
    outputs = Dense(1, activation = "sigmoid")(layer1)
    epochnum = 128
    minimizer = "sgd"
    cost = "mean_squared_error"

netPN = Model(inputs = inputs, outputs = outputs)
netPN.compile(optimizer = minimizer, loss = cost)
PNh = netPN.fit(x = X, y = PN, batch_size = 128, epochs = epochnum, verbose = 1, validation_split = 0.2, shuffle = True)
netPN.save_weights("Nets/v2/netPN" + str(architecture) + ".hdf5")

plt.figure(1)
plt.subplot(311)
plt.plot(PNh.history["loss"])
plt.plot(PNh.history["val_loss"])
plt.title("MSE Loss vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(["Training loss", "Testing loss"])

plt.figure(2)
plt.subplot(311)
x = [i for i in range(len(Data.dataX))]
yPNp = [netPN.predict(np.array([Data.dataX[i]]))[0][0] for i in range(len(Data.dataX))]
yPNo = [Data.dataPN[i][0] for i in range(len(Data.dataX))]
plt.plot(x, yPNp, label = "Predicted")
plt.plot(x, yPNo, label = "Observed")
plt.title("Predicted and Observed Values vs. Time")
plt.xlabel("Time")
plt.ylabel("P(PN > 10Kcells/L)")
plt.legend()

netCDA = Model(inputs = inputs, outputs = outputs)
netCDA.compile(optimizer = minimizer, loss = cost)
CDAh = netCDA.fit(x = X, y = CDA, batch_size = 128, epochs = epochnum, verbose = 1, validation_split = 0.2, shuffle = True)
netCDA.save_weights("Nets/v2/netCDA" + str(architecture) + ".hdf5")

plt.figure(1)
plt.subplot(312)
plt.plot(CDAh.history["loss"])
plt.plot(CDAh.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(["Training loss", "Testing loss"])

plt.figure(2)
plt.subplot(312)
x = [i for i in range(len(Data.dataX))]
yCDAp = [netCDA.predict(np.array([Data.dataX[i]]))[0][0] for i in range(len(Data.dataX))]
yCDAo = [Data.dataCDA[i][0] for i in range(len(Data.dataX))]
plt.plot(x, yCDAp, label = "Predicted")
plt.plot(x, yCDAo, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(CDA > 10pg/cell)")
plt.legend()

netPDA = Model(inputs = inputs, outputs = outputs)
netPDA.compile(optimizer = minimizer, loss = cost)
PDAh = netPDA.fit(x = X, y = PDA, batch_size = 128, epochs = epochnum, verbose = 1, validation_split = 0.2, shuffle = True)
netPDA.save_weights("Nets/v2/netPDA" + str(architecture) + ".hdf5")

plt.figure(1)
plt.subplot(313)
plt.plot(PDAh.history["loss"])
plt.plot(PDAh.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(["Training loss", "Testing loss"])
plt.savefig("Plots/v2/MSEvE" + str(architecture) + ".png")

plt.figure(2)
plt.subplot(313)
x = [i for i in range(len(Data.dataX))]
yPDAp = [netPDA.predict(np.array([Data.dataX[i]]))[0][0] for i in range(len(Data.dataX))]
yPDAo = [Data.dataPDA[i][0] for i in range(len(Data.dataX))]
plt.plot(x, yPDAp, label = "Predicted")
plt.plot(x, yPDAo, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(PDA > 500ng/L)")
plt.legend()
plt.savefig("Plots/v2/POvT" + str(architecture) + ".png")
