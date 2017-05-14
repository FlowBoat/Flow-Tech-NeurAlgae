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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import pkg_resources
#pkg_resources.require("keras == 1.1.0")
from keras.layers import Dense, LSTM, Dropout, regularizers
from keras.models import Sequential

np.random.seed(16)

print("Creating 0-day X data...")

dataX = open("Data/dataX.txt", "r")
X_0 = []
for i in dataX:
    line = i.split(" ")
    X_0.append([])
    for j in line:
        X_0[-1].append(float(j))
    X_0[-1] = np.array(X_0[-1])
X_0 = np.array(X_0)
dataX.close()

print("Creating 0-day PN data...")

dataPN = open("Data/dataPN.txt", "r")
PN_0 = []
for i in dataPN:
    line = i.split(" ")
    PN_0.append([])
    for j in line:
        PN_0[-1].append(float(j))
    PN_0[-1] = np.array(PN_0[-1])
PN_0 = np.array(PN_0)
dataPN.close()

print("Creating 0-day CDA data...")

dataCDA = open("Data/dataCDA.txt", "r")
CDA_0 = []
for i in dataCDA:
    line = i.split(" ")
    CDA_0.append([])
    for j in line:
        CDA_0[-1].append(float(j))
    CDA_0[-1] = np.array(CDA_0[-1])
CDA_0 = np.array(CDA_0)
dataCDA.close()

print("Creating 0-day PDA data...")

dataPDA = open("Data/dataPDA.txt", "r")
PDA_0 = []
for i in dataPDA:
    line = i.split(" ")
    PDA_0.append([])
    for j in line:
        PDA_0[-1].append(float(j))
    PDA_0[-1] = np.array(PDA_0[-1])
PDA_0 = np.array(PDA_0)
dataPDA.close()

print("Creating 0-day architecture...")

net0 = Sequential()
net0.add(Dense(128, input_shape = (9,), activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net0.add(Dropout(0.05))
net0.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net0.add(Dropout(0.05))
net0.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net0.add(Dropout(0.05))
net0.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net0.add(Dropout(0.05))
net0.add(Dense(1, activation = "sigmoid"))

print("Training 0-day PN network...")

netPN_0 = net0
netPN_0.compile(optimizer = "nadam", loss = "mean_squared_error")
PN_0h = netPN_0.fit(x = X_0, y = PN_0, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPN_0.save_weights("Nets/v2/v2.1/netPN_0.hdf5")
fPN_0 = open("Nets/v2/v2.1/netPN_0.json", "w")
fPN_0.write(netPN_0.to_json())
fPN_0.close()

print("Generating 0-day PN MSEvE graph...")

plt.figure(1)
plt.subplot(311)
plt.plot(PN_0h.history["loss"])
plt.plot(PN_0h.history["val_loss"])
plt.title("Immediate MSE Loss vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("PN MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 0-day PN POvT graph...")

plt.figure(2)
plt.subplot(311)
xPN_0 = [i for i in range(len(X_0))]
yPN_0p = [netPN_0.predict(np.array([X_0[i]]))[0][0] for i in range(len(X_0))]
yPN_0o = [PN_0[i][0] for i in range(len(X_0))]
plt.plot(xPN_0, yPN_0p, label = "Predicted")
plt.plot(xPN_0, yPN_0o, label = "Observed")
plt.title("Immediate Predicted and Observed Values vs. Time")
plt.xlabel("Time")
plt.ylabel("P(PN > 10Kcells/L)")
plt.legend()

print("Training 0-day CDA network...")

netCDA_0 = net0
netCDA_0.compile(optimizer = "nadam", loss = "mean_squared_error")
CDA_0h = netCDA_0.fit(x = X_0, y = CDA_0, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netCDA_0.save_weights("Nets/v2/v2.1/netCDA_0.hdf5")
fCDA_0 = open("Nets/v2/v2.1/netCDA_0.json", "w")
fCDA_0.write(netCDA_0.to_json())
fCDA_0.close()

print("Generating 0-day CDA MSEvE graph...")

plt.figure(1)
plt.subplot(312)
plt.plot(CDA_0h.history["loss"])
plt.plot(CDA_0h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("CDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 0-day CDA POvT graph...")

plt.figure(2)
plt.subplot(312)
xCDA_0 = [i for i in range(len(X_0))]
yCDA_0p = [netCDA_0.predict(np.array([X_0[i]]))[0][0] for i in range(len(X_0))]
yCDA_0o = [CDA_0[i][0] for i in range(len(X_0))]
plt.plot(xCDA_0, yCDA_0p, label = "Predicted")
plt.plot(xCDA_0, yCDA_0o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(CDA > 10pg/cell)")
plt.legend()

print("Training 0-day PDA network...")

netPDA_0 = net0
netPDA_0.compile(optimizer = "nadam", loss = "mean_squared_error")
PDA_0h = netPDA_0.fit(x = X_0, y = PDA_0, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPDA_0.save_weights("Nets/v2/v2.1/netPDA_0.hdf5")
fPDA_0 = open("Nets/v2/v2.1/netPDA_0.json", "w")
fPDA_0.write(netPDA_0.to_json())
fPDA_0.close()

print("Generating 0-day PDA MSEvE graph...")

plt.figure(1)
plt.subplot(313)
plt.plot(PDA_0h.history["loss"])
plt.plot(PDA_0h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("PDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])
plt.savefig("Plots/v2/v2.1/MSEvE_0.png")

print("Generating 0-day PDA POvT graph...")

plt.figure(2)
plt.subplot(313)
xPDA_0 = [i for i in range(len(X_0))]
yPDA_0p = [netPDA_0.predict(np.array([X_0[i]]))[0][0] for i in range(len(X_0))]
yPDA_0o = [PDA_0[i][0] for i in range(len(X_0))]
plt.plot(xPDA_0, yPDA_0p, label = "Predicted")
plt.plot(xPDA_0, yPDA_0o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(PDA > 500ng/L)")
plt.legend()
plt.savefig("Plots/v2/v2.1/POvT_0.png")

print("Creating prediction-mapped dataset...")

X = np.zeros((len(X_0), 12))
for i in range(len(X)):
    for j in range(len(X_0[i])):
        X[i][j] = X_0[i][j]
    X[i][9] = netPN_0.predict(np.array([X_0[i]]))[0][0]
    X[i][10] = netCDA_0.predict(np.array([X_0[i]]))[0][0]
    X[i][11] = netPDA_0.predict(np.array([X_0[i]]))[0][0]
    if i % 5000 == 0:
        print("[" + str(i) + "][" + str(j) + "] done")

print("Creating 15-minute X data...")

X_15 = np.zeros((len(X_0), 1, 12))
for i in range(len(X_15)):
    for j in range(len(X_15[i])):
        for k in range(len(X[(i + j) % len(X)])):
            X_15[i][j][k] = X[(i + j) % len(X)][k]
    if i % 5000 == 0:
        print("[" + str(i) + "][" + str(j) + "][" + str(k) + "] done")

print("Creating 15-minute PN data...")

PN_15 = np.zeros((len(PN_0), 1))
for i in range(len(PN_15)):
    PN_15[i] = PN_0[(i + 2 * 1) % len(PN_0)]

print("Creating 15-minute CDA data...")

CDA_15 = np.zeros((len(CDA_0), 1))
for i in range(len(CDA_15)):
    CDA_15[i] = CDA_0[(i + 2 * 1) % len(CDA_0)]

print("Creating 15-minute PDA data...")

PDA_15 = np.zeros((len(PDA_0), 1))
for i in range(len(PDA_15)):
    PDA_15[i] = PDA_0[(i + 2 * 1) % len(PDA_0)]

print("Creating 15-minute architecture...")

net15 = Sequential()
net15.add(LSTM(32, input_shape = (1, 12), return_sequences = False))
net15.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net15.add(Dropout(0.05))
net15.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net15.add(Dropout(0.05))
net15.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net15.add(Dropout(0.05))
net15.add(Dense(1, activation = "sigmoid"))

print("Training 15-minute PN network...")

netPN_15 = net15
netPN_15.compile(optimizer = "nadam", loss = "mean_squared_error")
PN_15h = netPN_15.fit(x = X_15, y = PN_15, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPN_15.save_weights("Nets/v2/v2.1/netPN_15.hdf5")
fPN_15 = open("Nets/v2/v2.1/netPN_15.json", "w")
fPN_15.write(netPN_15.to_json())
fPN_15.close()

print("Generating 15-minute PN MSEvE graph...")

plt.figure(3)
plt.subplot(311)
plt.plot(PN_15h.history["loss"])
plt.plot(PN_15h.history["val_loss"])
plt.title("15-min. MSE Loss vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("PN MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 15-minute PN POvT graph...")

plt.figure(4)
plt.subplot(311)
xPN_15 = [i for i in range(len(X_15))]
yPN_15p = [netPN_15.predict(np.array([X_15[i]]))[0][0] for i in range(len(X_15))]
yPN_15o = [PN_15[i][0] for i in range(len(X_15))]
plt.plot(xPN_15, yPN_15p, label = "Predicted")
plt.plot(xPN_15, yPN_15o, label = "Observed")
plt.title("15-min. Predicted and Observed Values vs. Time")
plt.xlabel("Time")
plt.ylabel("P(PN > 10Kcells/L)")
plt.legend()

print("Training 15-minute CDA network...")

netCDA_15 = net15
netCDA_15.compile(optimizer = "nadam", loss = "mean_squared_error")
CDA_15h = netCDA_15.fit(x = X_15, y = CDA_15, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netCDA_15.save_weights("Nets/v2/v2.1/netCDA_15.hdf5")
fCDA_15 = open("Nets/v2/v2.1/netCDA_15.json", "w")
fCDA_15.write(netCDA_15.to_json())
fCDA_15.close()

print("Generating 15-minute CDA MSEvE graph...")

plt.figure(3)
plt.subplot(312)
plt.plot(CDA_15h.history["loss"])
plt.plot(CDA_15h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("CDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 15-minute CDA POvT graph...")

plt.figure(4)
plt.subplot(312)
xCDA_15 = [i for i in range(len(X_15))]
yCDA_15p = [netCDA_15.predict(np.array([X_15[i]]))[0][0] for i in range(len(X_15))]
yCDA_15o = [CDA_15[i][0] for i in range(len(X_15))]
plt.plot(xCDA_15, yCDA_15p, label = "Predicted")
plt.plot(xCDA_15, yCDA_15o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(CDA > 10pg/cell)")
plt.legend()

print("Training 15-minute PDA network...")

netPDA_15 = net15
netPDA_15.compile(optimizer = "nadam", loss = "mean_squared_error")
PDA_15h = netPDA_15.fit(x = X_15, y = PDA_15, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPDA_15.save_weights("Nets/v2/v2.1/netPDA_15.hdf5")
fPDA_15 = open("Nets/v2/v2.1/netPDA_15.json", "w")
fPDA_15.write(netPDA_15.to_json())
fPDA_15.close()

print("Generating 15-minute PDA MSEvE graph...")

plt.figure(3)
plt.subplot(313)
plt.plot(PDA_15h.history["loss"])
plt.plot(PDA_15h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("PDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])
plt.savefig("Plots/v2/v2.1/MSEvE_15.png")

print("Generating 15-minute PDA POvT graph...")

plt.figure(4)
plt.subplot(313)
xPDA_15 = [i for i in range(len(X_15))]
yPDA_15p = [netPDA_15.predict(np.array([X_15[i]]))[0][0] for i in range(len(X_15))]
yPDA_15o = [PDA_15[i][0] for i in range(len(X_15))]
plt.plot(xPDA_15, yPDA_15p, label = "Predicted")
plt.plot(xPDA_15, yPDA_15o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(PDA > 500ng/L)")
plt.legend()
plt.savefig("Plots/v2/v2.1/POvT_15.png")

print("Creating 1-day X data...")

X_1 = np.zeros((len(X_0), 96, 12))
for i in range(len(X_1)):
    for j in range(len(X_1[i])):
        for k in range(len(X[(i + j) % len(X)])):
            X_1[i][j][k] = X[(i + j) % len(X)][k]
    if i % 5000 == 0:
        print("[" + str(i) + "][" + str(j) + "][" + str(k) + "] done")

print("Creating 1-day PN data...")

PN_1 = np.zeros((len(PN_0), 1))
for i in range(len(PN_1)):
    PN_1[i] = PN_0[(i + 2 * 96) % len(PN_0)]

print("Creating 1-day CDA data...")

CDA_1 = np.zeros((len(CDA_0), 1))
for i in range(len(CDA_1)):
    CDA_1[i] = CDA_0[(i + 2 * 96) % len(CDA_0)]

print("Creating 1-day PDA data...")

PDA_1 = np.zeros((len(PDA_0), 1))
for i in range(len(PDA_1)):
    PDA_1[i] = PDA_0[(i + 2 * 96) % len(PDA_0)]

print("Creating 1-day architecture...")

net1 = Sequential()
net1.add(LSTM(32, input_shape = (96, 12), return_sequences = False))
net1.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net1.add(Dropout(0.05))
net1.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net1.add(Dropout(0.05))
net1.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net1.add(Dropout(0.05))
net1.add(Dense(1, activation = "sigmoid"))

print("Training 1-day PN network...")

netPN_1 = net1
netPN_1.compile(optimizer = "nadam", loss = "mean_squared_error")
PN_1h = netPN_1.fit(x = X_1, y = PN_1, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPN_1.save_weights("Nets/v2/v2.1/netPN_1.hdf5")
fPN_1 = open("Nets/v2/v2.1/netPN_1.json", "w")
fPN_1.write(netPN_1.to_json())
fPN_1.close()

print("Generating 1-day PN MSEvE graph...")

plt.figure(5)
plt.subplot(311)
plt.plot(PN_1h.history["loss"])
plt.plot(PN_1h.history["val_loss"])
plt.title("1-day MSE Loss vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("PN MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 1-day PN POvT graph...")

plt.figure(6)
plt.subplot(311)
xPN_1 = [i for i in range(len(X_1))]
yPN_1p = [netPN_1.predict(np.array([X_1[i]]))[0][0] for i in range(len(X_1))]
yPN_1o = [PN_1[i][0] for i in range(len(X_1))]
plt.plot(xPN_1, yPN_1p, label = "Predicted")
plt.plot(xPN_1, yPN_1o, label = "Observed")
plt.title("1-day Predicted and Observed Values vs. Time")
plt.xlabel("Time")
plt.ylabel("P(PN > 10Kcells/L)")
plt.legend()

print("Training 1-day CDA network...")

netCDA_1 = net1
netCDA_1.compile(optimizer = "nadam", loss = "mean_squared_error")
CDA_1h = netCDA_1.fit(x = X_1, y = CDA_1, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netCDA_1.save_weights("Nets/v2/v2.1/netCDA_1.hdf5")
fCDA_1 = open("Nets/v2/v2.1/netCDA_1.json", "w")
fCDA_1.write(netCDA_1.to_json())
fCDA_1.close()

print("Generating 1-day CDA MSEvE graph...")

plt.figure(5)
plt.subplot(312)
plt.plot(CDA_1h.history["loss"])
plt.plot(CDA_1h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("CDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 1-day CDA POvT graph...")

plt.figure(6)
plt.subplot(312)
xCDA_1 = [i for i in range(len(X_1))]
yCDA_1p = [netCDA_1.predict(np.array([X_1[i]]))[0][0] for i in range(len(X_1))]
yCDA_1o = [CDA_1[i][0] for i in range(len(X_1))]
plt.plot(xCDA_1, yCDA_1p, label = "Predicted")
plt.plot(xCDA_1, yCDA_1o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(CDA > 10pg/cell)")
plt.legend()

print("Training 1-day PDA network...")

netPDA_1 = net1
netPDA_1.compile(optimizer = "nadam", loss = "mean_squared_error")
PDA_1h = netPDA_1.fit(x = X_1, y = PDA_1, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPDA_1.save_weights("Nets/v2/v2.1/netPDA_1.hdf5")
fPDA_1 = open("Nets/v2/v2.1/netPDA_1.json", "w")
fPDA_1.write(netPDA_1.to_json())
fPDA_1.close()

print("Generating 1-day PDA MSEvE graph...")

plt.figure(5)
plt.subplot(313)
plt.plot(PDA_1h.history["loss"])
plt.plot(PDA_1h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("PDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])
plt.savefig("Plots/v2/v2.1/MSEvE_1.png")

print("Generating 1-day PDA POvT graph...")

plt.figure(6)
plt.subplot(313)
xPDA_1 = [i for i in range(len(X_1))]
yPDA_1p = [netPDA_1.predict(np.array([X_1[i]]))[0][0] for i in range(len(X_1))]
yPDA_1o = [PDA_1[i][0] for i in range(len(X_1))]
plt.plot(xPDA_1, yPDA_1p, label = "Predicted")
plt.plot(xPDA_1, yPDA_1o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(PDA > 500ng/L)")
plt.legend()
plt.savefig("Plots/v2/v2.1/POvT_1.png")

print("Creating 3-day X data...")

X_3 = np.zeros((len(X_0), 288, 12))
for i in range(len(X_3)):
    for j in range(len(X_3[i])):
        for k in range(len(X[(i + j) % len(X)])):
            X_3[i][j][k] = X[(i + j) % len(X)][k]
    if i % 5000 == 0:
        print("[" + str(i) + "][" + str(j) + "][" + str(k) + "] done")

print("Creating 3-day PN data...")

PN_3 = np.zeros((len(PN_0), 1))
for i in range(len(PN_3)):
    PN_3[i] = PN_0[(i + 2 * 288) % len(PN_0)]

print("Creating 3-day CDA data...")

CDA_3 = np.zeros((len(CDA_0), 1))
for i in range(len(CDA_3)):
    CDA_3[i] = CDA_0[(i + 2 * 288) % len(CDA_0)]

print("Creating 3-day PDA data...")

PDA_3 = np.zeros((len(PDA_0), 1))
for i in range(len(PDA_3)):
    PDA_3[i] = PDA_0[(i + 2 * 288) % len(PDA_0)]

print("Creating 3-day architecture...")

net3 = Sequential()
net3.add(LSTM(32, input_shape = (288, 12), return_sequences = False))
net3.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net3.add(Dropout(0.05))
net3.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net3.add(Dropout(0.05))
net3.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net3.add(Dropout(0.05))
net3.add(Dense(1, activation = "sigmoid"))

print("Training 3-day PN network...")

netPN_3 = net3
netPN_3.compile(optimizer = "nadam", loss = "mean_squared_error")
PN_3h = netPN_3.fit(x = X_3, y = PN_3, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPN_3.save_weights("Nets/v2/v2.1/netPN_3.hdf5")
fPN_3 = open("Nets/v2/v2.1/netPN_3.json", "w")
fPN_3.write(netPN_3.to_json())
fPN_3.close()

print("Generating 3-day PN MSEvE graph...")

plt.figure(7)
plt.subplot(311)
plt.plot(PN_3h.history["loss"])
plt.plot(PN_3h.history["val_loss"])
plt.title("3-day MSE Loss vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("PN MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 3-day PN POvT graph...")

plt.figure(8)
plt.subplot(311)
xPN_3 = [i for i in range(len(X_3))]
yPN_3p = [netPN_3.predict(np.array([X_3[i]]))[0][0] for i in range(len(X_3))]
yPN_3o = [PN_3[i][0] for i in range(len(X_3))]
plt.plot(xPN_3, yPN_3p, label = "Predicted")
plt.plot(xPN_3, yPN_3o, label = "Observed")
plt.title("3-day Predicted and Observed Values vs. Time")
plt.xlabel("Time")
plt.ylabel("P(PN > 10Kcells/L)")
plt.legend()

print("Training 3-day CDA network...")

netCDA_3 = net3
netCDA_3.compile(optimizer = "nadam", loss = "mean_squared_error")
CDA_3h = netCDA_3.fit(x = X_3, y = CDA_3, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netCDA_3.save_weights("Nets/v2/v2.1/netCDA_3.hdf5")
fCDA_3 = open("Nets/v2/v2.1/netCDA_3.json", "w")
fCDA_3.write(netCDA_3.to_json())
fCDA_3.close()

print("Generating 3-day CDA MSEvE graph...")

plt.figure(7)
plt.subplot(312)
plt.plot(CDA_3h.history["loss"])
plt.plot(CDA_3h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("CDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 3-day CDA POvT graph...")

plt.figure(8)
plt.subplot(312)
xCDA_3 = [i for i in range(len(X_3))]
yCDA_3p = [netCDA_3.predict(np.array([X_3[i]]))[0][0] for i in range(len(X_3))]
yCDA_3o = [CDA_3[i][0] for i in range(len(X_3))]
plt.plot(xCDA_3, yCDA_3p, label = "Predicted")
plt.plot(xCDA_3, yCDA_3o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(CDA > 10pg/cell)")
plt.legend()

print("Training 3-day PDA network...")

netPDA_3 = net3
netPDA_3.compile(optimizer = "nadam", loss = "mean_squared_error")
PDA_3h = netPDA_3.fit(x = X_3, y = PDA_3, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPDA_3.save_weights("Nets/v2/v2.1/netPDA_3.hdf5")
fPDA_3 = open("Nets/v2/v2.1/netPDA_3.json", "w")
fPDA_3.write(netPDA_3.to_json())
fPDA_3.close()

print("Generating 3-day PDA MSEvE graph...")

plt.figure(7)
plt.subplot(313)
plt.plot(PDA_3h.history["loss"])
plt.plot(PDA_3h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("PDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])
plt.savefig("Plots/v2/v2.1/MSEvE_3.png")

print("Generating 3-day PDA POvT graph...")

plt.figure(8)
plt.subplot(313)
xPDA_3 = [i for i in range(len(X_3))]
yPDA_3p = [netPDA_3.predict(np.array([X_3[i]]))[0][0] for i in range(len(X_3))]
yPDA_3o = [PDA_3[i][0] for i in range(len(X_3))]
plt.plot(xPDA_3, yPDA_3p, label = "Predicted")
plt.plot(xPDA_3, yPDA_3o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(PDA > 500ng/L)")
plt.legend()
plt.savefig("Plots/v2/v2.1/POvT_3.png")

print("Creating 7-day X data...")

X_7 = np.zeros((len(X_0), 672, 12))
for i in range(len(X_7)):
    for j in range(len(X_7[i])):
        for k in range(len(X[(i + j) % len(X)])):
            X_7[i][j][k] = X[(i + j) % len(X)][k]
    if i % 5000 == 0:
        print("[" + str(i) + "][" + str(j) + "][" + str(k) + "] done")

print("Creating 7-day PN data...")

PN_7 = np.zeros((len(PN_0), 1))
for i in range(len(PN_7)):
    PN_7[i] = PN_0[(i + 2 * 672) % len(PN_0)]

print("Creating 7-day CDA data...")

CDA_7 = np.zeros((len(CDA_0), 1))
for i in range(len(CDA_7)):
    CDA_7[i] = CDA_0[(i + 2 * 672) % len(CDA_0)]

print("Creating 7-day PDA data...")

PDA_7 = np.zeros((len(PDA_0), 1))
for i in range(len(PDA_7)):
    PDA_7[i] = PDA_0[(i + 2 * 672) % len(PDA_0)]

print("Creating 7-day architecture...")

net7 = Sequential()
net7.add(LSTM(32, input_shape = (672, 12), return_sequences = False))
net7.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net7.add(Dropout(0.05))
net7.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net7.add(Dropout(0.05))
net7.add(Dense(128, activation = "relu", activity_regularizer = regularizers.l2(0.0001)))
net7.add(Dropout(0.05))
net7.add(Dense(1, activation = "sigmoid"))

print("Training 7-day PN network...")

netPN_7 = net7
netPN_7.compile(optimizer = "nadam", loss = "mean_squared_error")
PN_7h = netPN_7.fit(x = X_7, y = PN_7, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPN_7.save_weights("Nets/v2/v2.1/netPN_7.hdf5")
fPN_7 = open("Nets/v2/v2.1/netPN_7.json", "w")
fPN_7.write(netPN_7.to_json())
fPN_7.close()

print("Generating 7-day PN MSEvE graph...")

plt.figure(9)
plt.subplot(311)
plt.plot(PN_7h.history["loss"])
plt.plot(PN_7h.history["val_loss"])
plt.title("7-day MSE Loss vs. Training Epoch")
plt.xlabel("Epoch")
plt.ylabel("PN MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 7-day PN POvT graph...")

plt.figure(10)
plt.subplot(311)
xPN_7 = [i for i in range(len(X_7))]
yPN_7p = [netPN_7.predict(np.array([X_7[i]]))[0][0] for i in range(len(X_7))]
yPN_7o = [PN_7[i][0] for i in range(len(X_7))]
plt.plot(xPN_7, yPN_7p, label = "Predicted")
plt.plot(xPN_7, yPN_7o, label = "Observed")
plt.title("7-day Predicted and Observed Values vs. Time")
plt.xlabel("Time")
plt.ylabel("P(PN > 10Kcells/L)")
plt.legend()

print("Training 7-day CDA network...")

netCDA_7 = net7
netCDA_7.compile(optimizer = "nadam", loss = "mean_squared_error")
CDA_7h = netCDA_7.fit(x = X_7, y = CDA_7, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netCDA_7.save_weights("Nets/v2/v2.1/netCDA_7.hdf5")
fCDA_7 = open("Nets/v2/v2.1/netCDA_7.json", "w")
fCDA_7.write(netCDA_7.to_json())
fCDA_7.close()

print("Generating 7-day CDA MSEvE graph...")

plt.figure(9)
plt.subplot(312)
plt.plot(CDA_7h.history["loss"])
plt.plot(CDA_7h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("CDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])

print("Generating 7-day CDA POvT graph...")

plt.figure(10)
plt.subplot(312)
xCDA_7 = [i for i in range(len(X_7))]
yCDA_7p = [netCDA_7.predict(np.array([X_7[i]]))[0][0] for i in range(len(X_7))]
yCDA_7o = [CDA_7[i][0] for i in range(len(X_7))]
plt.plot(xCDA_7, yCDA_7p, label = "Predicted")
plt.plot(xCDA_7, yCDA_7o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(CDA > 10pg/cell)")
plt.legend()

print("Training 7-day PDA network...")

netPDA_7 = net7
netPDA_7.compile(optimizer = "nadam", loss = "mean_squared_error")
PDA_7h = netPDA_7.fit(x = X_7, y = PDA_7, batch_size = 32, epochs = 32, validation_split = 0.2, verbose = 1, shuffle = True)
netPDA_7.save_weights("Nets/v2/v2.1/netPDA_7.hdf5")
fPDA_7 = open("Nets/v2/v2.1/netPDA_7.json", "w")
fPDA_7.write(netPDA_7.to_json())
fPDA_7.close()

print("Generating 7-day PDA MSEvE graph...")

plt.figure(9)
plt.subplot(313)
plt.plot(PDA_7h.history["loss"])
plt.plot(PDA_7h.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("PDA MSE Loss")
plt.legend(["Training loss", "Testing loss"])
plt.savefig("Plots/v2/v2.1/MSEvE_7.png")

print("Generating 7-day PDA POvT graph...")

plt.figure(10)
plt.subplot(313)
xPDA_7 = [i for i in range(len(X_7))]
yPDA_7p = [netPDA_7.predict(np.array([X_7[i]]))[0][0] for i in range(len(X_7))]
yPDA_7o = [PDA_7[i][0] for i in range(len(X_7))]
plt.plot(xPDA_7, yPDA_7p, label = "Predicted")
plt.plot(xPDA_7, yPDA_7o, label = "Observed")
plt.xlabel("Time")
plt.ylabel("P(PDA > 500ng/L)")
plt.legend()
plt.savefig("Plots/v2/v2.1/POvT_7.png")
