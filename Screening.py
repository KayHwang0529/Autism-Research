import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

#reading the data
data = pd.read_csv("data/Autism_Data.arff")

#cleaning data
data = data.rename(columns = {'austim':'family member with PDD','jundice':'jaundice','contry_of_res':'country_of_res'})

data['age'] = data['age'].apply(lambda x: np.nan if x == '?' else int(x))
data['gender'] = data['gender'].map({'m':1,'f':0})
data['jaundice'] = data['jaundice'].map({'yes':1,'no':0})
data['family member with PDD'] = data['family member with PDD'].map({'yes':1,'no':0})
data['used_app_before'] = data['used_app_before'].map({'yes':1,'no':0})
data['Class/ASD'] = data['Class/ASD'].map({'YES':1,'NO':0})
data['ethnicity'] = data["ethnicity"].map({"White-European": 1, "Others": 2, "Asian": 3,"Middle Eastern": 4, "Black": 5,"South Asian": 6, "Hispanic": 7, "Pasifika":8, "Turkish": 9} )
del data['country_of_res']
del data['age_desc']
del data['relation']

data = data.dropna()

print(data.info)

sns.heatmap(data.corr(),annot=True)
plt.title('Heatmap of Variable Correlations')

#ASD coorellates highly with A5, A6, and A9 scores

#______seperates classification column from the data____

results = data["Class/ASD"]
data = data.drop("Class/ASD", axis = 1)

data = data.astype(int)


dataArray = data.to_numpy()
resultsArray = results.to_numpy()


X, Y = np.array(dataArray), np.array(resultsArray)

#______________Normalizes data_______________

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)

#______________Tensorflow model______________

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [tf.keras.Input(shape=(17,)),Dense(25, activation='relu', name = 'layer1'),Dense(15, activation='relu', name = 'layer2'),Dense(1, activation='sigmoid', name = 'layer3')]
)


#specifices loss function (avg. gives cost function)________________
model.compile(
    loss = BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01), #Adam is a safe optimizer choice
    #for softmax regression use loss = SparseCategoricalCrossEntropy(from_logits=True)
)

#________preforms back propagation 100 times____________________
model.fit(Xn,Y,epochs=100)


#___Test______________________________________
dataArray = norm_l(dataArray)
predictions = model.predict(dataArray)
#for softmax you have to update the predict function to take in logit
print("predictions = \n", predictions)

# To convert the probabilities to a decision, we apply a threshold:

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")


#_______________ checking accuracy ___________________________
m = BinaryAccuracy()
m.update_state(results,predictions)
print(m.result().numpy())


