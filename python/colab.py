import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

#Load JSON files with datasets
#non gesture
nonGesturesFile = open('/content/NonGestures.json')
nonGesturesJson = json.load(nonGesturesFile)

#Victoria gesture
victoriaDatasetFile = open('/content/VictoriaDataset.json')
victoriaDatasetJson = json.load(victoriaDatasetFile)

victoriaTestFile = open('/content/VictoriaTest.json')
victoriaTestJson = json.load(victoriaTestFile)

#OK gesture
okDatasetFile = open('/content/OkDataset.json')
okDatasetJson = json.load(okDatasetFile)

okTestFile = open('/content/OkTest.json')
okTestJson = json.load(okTestFile)

#Prepare dataset x=>y
xDataset = []
yDataset = []

#0 for non gestures
for el in nonGesturesJson:
  xDataset.append(el)
  yDataset.append(0)

#1 for Victoria gesture
for el in victoriaDatasetJson:
  xDataset.append(el)
  yDataset.append(1)

#2 for OK gesture
for el in okDatasetJson:
  xDataset.append(el)
  yDataset.append(2)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(21, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(3)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(xDataset, yDataset, epochs=15)

#Test model - prepare and evaluate
#Prepare dataset x=>y
xTest = []
yTest = []

#0 for non gestures

#1 for Victoria gesture
for el in victoriaTestJson:
  xTest.append(el)
  yTest.append(1)

#2 for OK gesture
for el in okTestJson:
  xTest.append(el)
  yTest.append(2)

model.evaluate(xTest,  yTest, verbose=2)

#Create probability model
probabilityModel = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

#test prediction OK then Victoria file
testPredictionFile = open('/content/testPred.json')
testPredictionJson1 = json.load(testPredictionFile)

predictions1 = probabilityModel.predict(testPredictionJson1)
for i in range(len(predictions1)):
  print(i, np.argmax(predictions1[i]))

#second test
testPredictionFile = open('/content/testPrediction.json')
testPredictionJson2 = json.load(testPredictionFile)

predictions1 = probabilityModel.predict(testPredictionJson1)
for i in range(len(predictions1)):
  print(i, np.argmax(predictions1[i]))

pip install tensorflowjs

import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, 'modeljs')