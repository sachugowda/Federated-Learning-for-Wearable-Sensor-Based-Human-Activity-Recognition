import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
import keras
from sklearn.utils import resample
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from trans import *

sweep_config = {
  'method': 'grid',
  'metric': {
    'goal': 'maximize',
    'name': 'val_accuracy'
  },
  'parameters': {
      'epochs': {
        'value': 50
      },
      'num_layers': {
        'value': 3
      },
      'embed_layer_size': {
        'value': 128
      },
      'fc_layer_size': {
        'value': 256
      },
      'num_heads': {
        'value': 6
      },
      'dropout': {
        'value': 0.1
      },
      'attention_dropout': {
        'value': 0.1
      },
      'optimizer': {
        'value': 'adam'
      },
      'amsgrad': {
        'value': False
      },
      'label_smoothing': {
        'value': 0.1
      },
      'learning_rate': {
        'value': 1e-3
      },
      #'weight_decay': {
      #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
      #},
      'warmup_steps': {
        'value': 10
      },
      'batch_size': {
        'value': 64
      },
      'global_clipnorm': {
        'value': 3.0
      },
    }
}
sweep_id = wandb.sweep(sweep_config, project="imu-transformer")


tf.random.set_seed(42)

df = pd.read_csv('data/mhealth_raw_data.csv')

df_majority = df[df.Activity==0]
df_minorities = df[df.Activity!=0]
df_majority_downsampled = resample(df_majority,n_samples=30000, random_state=42)
df = pd.concat([df_majority_downsampled, df_minorities])

#Dropping feature have data outside 98% confidence interval
df1 = df.copy()

for feature in df1.columns[:-2]:
  lower_range = np.quantile(df[feature],0.01)
  upper_range = np.quantile(df[feature],0.99)
  print(feature,'range:',lower_range,'to',upper_range)

  df1 = df1.drop(df1[(df1[feature]>upper_range) | (df1[feature]<lower_range)].index, axis=0)
  
  label_map = {
    0: 'Nothing',
    1: 'Standing still',  
    2: 'Sitting and relaxing', 
    3: 'Lying down',  
    4: 'Walking',  
    5: 'Climbing stairs',  
    6: 'Waist bends forward',
    7: 'Frontal elevation of arms', 
    8: 'Knees bending (crouching)', 
    9: 'Cycling', 
    10: 'Jogging', 
    11: 'Running', 
    12: 'Jump front & back' 
}

df1 = df1[(df1['subject'] != 'subject3') & (df1['subject'] != 'subject4') & (df1['subject'] != 'subject5') & (df1['subject'] != 'subject6')  & (df1['subject'] != 'subject7') & (df1['subject'] != 'subject8')]
#y = df1.Activity
#X = df1.drop(['Activity','subject'], axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

#spliting data into train and test set
train = df1[(df1['subject'] != 'subject9') & (df1['subject'] != 'subject10')]
test = df1.drop(train.index, axis=0)


allmeans=[]
allstds=[]
#I HAVE TAKEN STANDARD MEAN AND STANDARD DEVIATION OF EACH DATA (d1,d2,d3,d4,d5,d6,d7,d8) STORED IN ONE LIST
#THEN I HAVE TAKEN THE MEAN OF THOSE TWO
def get_data(data):
    X_train = data.drop(['Activity','subject'],axis=1)
    y_train = data['Activity']
    scaler=StandardScaler()
    X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
    allmeans.append(scaler.mean_)
    allstds.append(scaler.scale_)
    return X_train, y_train

X_train, y_train = get_data(df1)

meant=np.mean(allmeans,axis=0)
stdt=np.mean(allstds,axis=0)

X_test = test.drop(['Activity','subject'],axis=1)
X_test1=np.array(X_test)
X_test1=(X_test1-meant)/stdt
X_test=pd.DataFrame(X_test1,columns=X_test.columns)
y_test = test['Activity']




#X_train = train.drop(['Activity','subject'],axis=1)
#y_train = train['Activity']
#X_test = test.drop(['Activity','subject'],axis=1)
#y_test = test['Activity']
  
#function to create time series datset for seuence modeling
def create_dataset(X, y, time_steps, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(x)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1) 
    

# Load dataset
X_train,y_train = create_dataset(X_train, y_train, 1, step=50)
X_test,y_test = create_dataset(X_test, y_test, 1, step=50)

# Generate new model

model = Transformer(
        num_layers=3,
        embed_dim=128,
        mlp_dim=256,
        num_heads=6,
        num_classes=13,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        )

        # adapt on training dataset - must be before model.compile !!!
        #model.input_norm.adapt(X_train, batch_size=config.batch_size)
        #print(model.input_norm.variables)

        

model.compile(
            loss=smoothed_sparse_categorical_crossentropy(label_smoothing=0.1),
            optimizer='adam',
            metrics=["accuracy"],
        )        


  
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    
        def get_parameters(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            r = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
            hist = r.history
            print("Fit history : " ,hist)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print("Eval accuracy : ", accuracy)
            return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)
