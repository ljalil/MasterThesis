# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:51:08 2020

@author: Abdeljalil Letrache <abdeljalilletrache@outlook.com>
"""
#%% Load Basic Libraries
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

os.environ['PYTHONHASHSEED']=str(1)
#%% Load Sci-Kit Utitilies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#%% Load Keras Utilities
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical

#%% Matplotlib LaTeX Settings
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('axes', titlesize=12)
plt.rc('text', usetex=True)

#%% CMAPSS class
class CMAPSS():
    def __init__(self, path_to_data, fd_number):
        self.columns = ['unit','cycle','op1','op2','op3']+['sensor_{}'.format(str(i).zfill(2)) for i in range(1,24)]
        self.sensors = ['Total temperature at fan inlet',
           'Total temperature at LPC outlet',
           'Total temperature at HPC outlet',
           'Total temperature at LPT',
           'P2 Pressure at fan inlet',
           'Total pressure in bypass-duct',
           'Total pressure at HPC outlet',
           'Physical fan speed',
           'Physical core speed',
           'Engine pressure ratio (P50/P2)',
           'Static pressure at HPC outlet',
           'Ratio of fuel flow to Ps30',
           'Corrected fan speed',
           'Corrected core speed',
           'Bypass Ratio',
           'Burner fuel-air ratio',
           'Bleed Enthalpy',
           'Demanded fan speed',
           'Demanded corrected fan speed',
           'HPT coolant bleed',
           'LPT coolant bleed']
        
        self.path_to_file  = os.path.join(path_to_data, 'train_FD00'+str(fd_number)+'.txt')
        self.data = pd.read_csv(self.path_to_file, sep=' ', names=self.columns)
        self.data = self.data.drop(['sensor_22','sensor_23'], axis=1)
        self.rul = array([])
        self.random_test_units = random.uniform(1, self.data['unit'].max(), 4).astype('int32')
        
        self.sequence_length = 50
        self.start_of_failure = 120
        self.constant_rul = 130
        
    def plot_sensors(self, unit_number, sensors_to_plot, save_to_file = ""):
        fig, axes = plt.subplots(2,2,figsize=(6,2.5), sharex=True)
        
        unit = self.data[self.data['unit']==unit_number]
        
        for axis in enumerate(axes.flatten()):
            sensor_to_plot = self.data.columns[ sensors_to_plot[axis[0]] +3 ]
            axis[1].plot(unit[sensor_to_plot],c='k', lw=1)
            axis[1].set_title(sensors[ sensors_to_plot[axis[0]] ])
            axis[1].grid(ls=':')
        fig.tight_layout(h_pad=1)
        
        if save_to_file is not "":
            fig.savefig(save_to_file)
        
    def plot_units_pca(self, units_numbers, save_to_file = ""):
        units = []
        for i in range(len(units_numbers)):
            unit = df[df['unit']==units_numbers[i]]
            unit = unit[unit.columns[5:]]
            units.append(unit)
            
        pca = PCA(n_components=2).fit(concatenate(units, axis=0))
        
        for unit in enumerate(units):
            temp = pca.transform(unit[1])
            units[unit[0]]=temp
            
        scaler = StandardScaler().fit(concatenate(units, axis=0))
    
        fig, axes = plt.subplots(2,2,figsize=(5,5), sharex=True, sharey=True)
        
        for axis_units in zip(axes.flatten(), units):
            scaled_unit = scaler.transform(axis_units[1])
            x = scaled_unit[:,0]
            y = scaled_unit[:,1]
            sn.scatterplot(x, y, ax=axis_units[0], hue=arange(x.shape[0]),palette='inferno', size=4, linewidth=0)
            axis_units[0].get_legend().remove()
            
        for axis in enumerate(axes.flatten()):
            axis[1].set_title('Engine {}'.format(units_numbers[axis[0]]))
            axis[1].grid(ls=':')

        fig.tight_layout()
        
        if save_to_file is not "":
            fig.savefig(save_to_file)
            
    def RUL_modeler(self, life : int, kind : str, classes = 5):
        if kind == 'linear':
            return flip(arange(life)).reshape(-1,1)
        
        elif kind == 'nonlinear':
            x = array([0, 1.4*life/2, life])
            y = array([life, 1.2*life/2, 0])
        
            polynom = poly1d(polyfit(x,y,3))
        
            return polynom(arange(0,life)).reshape(-1,1)
        
        elif kind == 'piecewise':
            x = concatenate(( self.constant_rul*ones((self.start_of_failure,1)), linspace(self.constant_rul,0,life-self.start_of_failure).reshape(-1,1)), axis=0)
            return x
        
        elif kind == 'classification':
            return linspace(0, classes, life, endpoint = False ).astype('int32' ).reshape(-1,1)
        
    def calculate_RUL_all_units(self, kind : str):
        self.rul = array([])
        all_units_lengths = self.data.groupby('unit')['cycle'].max()
        
        for unit_length in all_units_lengths:
            self.rul = append(self.rul, self.RUL_modeler(unit_length, kind))
            
    def calculate_RUL_from_dataframe(self, df, kind : str):
        df_rul = array([])
        all_units_lengths = df.groupby('unit')['cycle'].max()
        
        for unit_length in all_units_lengths:
            df_rul = append(df_rul, self.RUL_modeler(unit_length, kind))
        
        return df_rul
        
    def construct_binary_classification_data(self, good_faulty_threshould = 30):
        x = zeros((0,26))
        y = zeros((0,1))
        
        for group_name, group_df in self.data.groupby('unit'):
            first_rows = group_df[:good_faulty_threshould]
            last_rows = group_df[-good_faulty_threshould:]
            
            x = concatenate((x, first_rows, last_rows), axis=0)
            y = concatenate((y, zeros((good_faulty_threshould,1)), ones((good_faulty_threshould,1))), axis=0)
            
        return x, y
    
    def dataset_statistics(self):
        units_number = int(self.data.iloc[-1]['unit'])
        max_length = self.data.groupby('unit')['cycle'].max().max()
        average_length = around(self.data.groupby('unit')['cycle'].max().mean(),2)
        min_length = self.data.groupby('unit')['cycle'].max().min()
        
        return dict({'units' : units_number, 'max': max_length, 'average' : average_length, 'min':min_length})
 
    def get_regression_train_test(self):
        self.regression_train_data = self.data.copy()
        self.regression_test_data = pd.DataFrame(columns = self.data.columns)
        
        for test_unit in self.random_test_units:
            cond = self.regression_train_data.unit == test_unit 
            self.regression_test_data = pd.concat((self.regression_test_data, self.regression_train_data[cond]), axis=0)
            
        self.regression_train_data.drop(self.regression_test_data.index, axis=0, inplace=True)
            
        return self.regression_train_data, self.regression_test_data
    
    def get_random_test_unit(self, id: int):
        return self.data[self.data['unit'] == self.random_test_units[id]]
        
    def mask_unit_for_lstm(self, unit : int, mask_value = 0):
        unit_values = self.data[self.data['unit'] == unit].values[:,2:]
        #Different units have different lines count, to shape data for use with LSTM, data must be shaped (batch_size, timesteps, input_dim), sequence_length is set to 20, most of the units lines can't be divided by sequence length for perfectly reshaping the unit, so it must be padded (with zeros)
        missing_lines = int(ceil(unit_values.shape[0]/self.sequence_length)*self.sequence_length-unit_values.shape[0])
        unit_values = concatenate((unit_values, mask_value*ones((missing_lines, unit_values.shape[1]))), axis=0)
        
        return unit_values
    
    def unit_rul_for_lstm(self, unit : int, mask_value = 0, kind = 'nonlinear'):
        unit_length = int(self.data[self.data['unit']==unit]['cycle'].max())
        missing_lines = int(ceil(unit_length/self.sequence_length)*self.sequence_length-unit_length)
        rul = self.RUL_modeler(unit_length, kind)
        rul = concatenate((rul, mask_value*ones((missing_lines,1))))
        return rul.reshape(-1, self.sequence_length)
        
    
    def get_lstm_train_test(self, mask_value=0, kind='nonlinear'):
        self.lstm_train_data = zeros((0, self.sequence_length, 24))
        self.lstm_test_data = zeros((0, self.sequence_length, 24))
        self.lstm_train_rul = zeros((0, self.sequence_length))
        self.lstm_test_rul = zeros((0, self.sequence_length))
        units_number = int(self.data.iloc[-1]['unit'])
        
        for unit_number in range(1,units_number+1):
            if not (self.random_test_units == unit_number).any():
                unit = self.mask_unit_for_lstm(unit_number, mask_value)
                self.lstm_train_data = concatenate((self.lstm_train_data, unit.reshape(-1, self.sequence_length, 24)) ,axis=0)

                rul = self.unit_rul_for_lstm(unit_number, mask_value, kind=kind)
                self.lstm_train_rul = concatenate((self.lstm_train_rul, rul))
            else:
                unit = self.mask_unit_for_lstm(unit_number, mask_value)
                self.lstm_test_data = concatenate((self.lstm_test_data, unit.reshape(-1, self.sequence_length, 24)) ,axis=0)
                rul = self.unit_rul_for_lstm(unit_number, mask_value, kind=kind)
                self.lstm_test_rul = concatenate((self.lstm_test_rul, rul))
             
        return self.lstm_train_data, self.lstm_test_data, self.lstm_train_rul, self.lstm_test_rul
    
    def get_lstm_test_unit(self, unit : int, mask_value=0, kind='nonlinear'):
        unit_data = self.mask_unit_for_lstm(self.random_test_units[unit], mask_value)
        unit_data = unit_data.reshape(-1, self.sequence_length, 24)
        rul = self.unit_rul_for_lstm(self.random_test_units[unit],mask_value= mask_value, kind=kind)
        
        return unit_data, rul
        
        
#%% Creating CMAPSS object with FD_001
cmapss1 = CMAPSS('C:/Users/abdel/Documents/Workspace/NASA', 1)
cmapss2 = CMAPSS('C:/Users/abdel/Documents/Workspace/NASA', 2)
cmapss3 = CMAPSS('C:/Users/abdel/Documents/Workspace/NASA', 3)
cmapss4 = CMAPSS('C:/Users/abdel/Documents/Workspace/NASA', 4)

#%% Calculate RUL

cmapss.calculate_RUL_all_units('nonlinear')

#%% Prepare Classification Data
x_classification, y_classification = cmapss.construct_binary_classification_data(25)
x_classification2, y_classification2 = cmapss2.construct_binary_classification_data(25)
x_classification3, y_classification3 = cmapss3.construct_binary_classification_data(25)
x_classification4, y_classification4 = cmapss4.construct_binary_classification_data(25)
#%%
x_classification = concatenate((x_classification,x_classification2,x_classification3,x_classification4), axis=0)
y_classification = concatenate((y_classification,y_classification2,y_classification3,y_classification4), axis=0)

#%%
x_classification_train, x_classification_test, y_classification_train, y_classification_test = train_test_split(x_classification[:,2:], y_classification, test_size=0.2)


#%% Fully-Connected : Classification
model_fc_clf = Sequential()
model_fc_clf.add( Dense(8, activation='relu', input_shape=(24,), name='Dense1') )
model_fc_clf.add( Dense(4, activation='relu', name='Dense2') )
model_fc_clf.add( Dense(1, activation='sigmoid', name='Dense3') )
model_fc_clf.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')

fc_classifcation_history = model_fc_clf.fit(x_classification_train, y_classification_train, epochs=200, batch_size=32, validation_split=.2)

#%% Evaluate Model : Fully-Connected Classification
score, acc = model_fc_clf.evaluate(x_classification_test, y_classification_test)
model_fc_clf_pred = model_fc_clf.predict(x_classification_test)
model_fc_clf_pred = around(model_fc_clf_pred)

fc_clf_precision = precision_score( y_classification_test, model_fc_clf_pred )
fc_clf_recall = recall_score( y_classification_test, model_fc_clf_pred )
fc_clf_f1 = f1_score( y_classification_test, model_fc_clf_pred )
fc_clf_roc_auc = roc_auc_score( y_classification_test, model_fc_clf_pred )

print('Score (binary crossentropy): {}'.format(around(score,4)))
print('Accuracy: {}%'.format(around(acc*100,2)))
print('Precision Score: {}'.format( around(fc_clf_precision,2) ))
print('Recall Score: {}'.format( around(fc_clf_recall,2) ))
print('F1 Score: {}'.format( around(fc_clf_f1,2) ))
print('ROC AUC Score: {}'.format( around(fc_clf_roc_auc,4) ))

#%% Plot Fully-Connected Training : Classification
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(classification_history_fc['loss'],c='k', lw=1)
axes[0].plot(classification_history_fc['val_loss'],'--',c='k', lw=1)
axes[0].legend(['Training','Validation'], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Loss (binary corssentropy)')

axes[1].plot(array(classification_history_fc['acc'])*100,c='k', lw=1)
axes[1].plot(array(classification_history_fc['val_acc'])*100,'--',c='k', lw=1)
axes[1].legend(['Training','Validation'], loc='lower right', ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy (\%)')

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_classification_training.pdf')

#%% Plot Classification History [French]
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(classification_history_fc['loss'],c='k', lw=1)
axes[0].plot(classification_history_fc['val_loss'],'--',c='k', lw=1)
axes[0].legend(["Entrînement","Validation"], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Perte (binary corssentropy)')

axes[1].plot(array(classification_history_fc['acc'])*100,c='k', lw=1)
axes[1].plot(array(classification_history_fc['val_acc'])*100,'--',c='k', lw=1)
axes[1].legend(["Entraînement",'Validation'], loc='lower right', ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epoques')
axes[1].set_ylabel('Précision (\%)')

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_classification_training_fr.pdf')

#%% Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_classification_test, model_fc_clf.predict(x_classification_test))

#%% Plot ROC Curve
fig, axis = plt.subplots(1,1, figsize=(4,3))
axis.plot(fpr,tpr,c='k', lw=1)
axis.plot([0, 1], [0, 1], 'k--', lw='.8')
axis.set_xlabel('False Positive Rate')
axis.set_ylabel('True Positive Rate')
axis.grid(ls=':')
axis.axis([0, 1, 0, 1])
axis.legend(['ROC Curve (AUC={})'.format(around(fc_clf_roc_auc,4))],framealpha=.5)
fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_classification_roc.pdf')

#%% Plot ROC Curve [French]
fig, axis = plt.subplots(1,1, figsize=(4,3))
axis.plot(fpr,tpr,c='k', lw=1)
axis.plot([0, 1], [0, 1], 'k--', lw='.8')
axis.set_xlabel('Taux de faux positif')
axis.set_ylabel('Taux de vrai positif')
axis.grid(ls=':')
axis.axis([0, 1, 0, 1])
axis.legend(['Courbe ROC (AUC={})'.format(around(fc_clf_roc_auc,4))],framealpha=.5)
fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_classification_roc_fr.pdf')

#%% Prepare Regression Data
x_regression_train, x_regression_test = cmapss1.get_regression_train_test()

y_regression_train = cmapss1.calculate_RUL_from_dataframe(x_regression_train, kind='nonlinear')
y_regression_test = cmapss1.calculate_RUL_from_dataframe(x_regression_test, kind='nonlinear')

x_regression_train = x_regression_train.values[:,2:]
x_regression_test = x_regression_test.values[:,2:]

#%% Fully-Connected : Regression
model_fc_reg = Sequential()
model_fc_reg.add( Dense(32, activation='relu', input_shape=(24,), name='Dense1') )
model_fc_reg.add( Dense(16, activation='relu', name='Dense2') )
model_fc_reg.add( Dense(8, activation='relu', name='Dense3') )
model_fc_reg.add( Dense(1, name='Dense4') )
model_fc_reg.compile(loss='mean_squared_error', metrics=['mae'], optimizer='adam')

#%% Train Fully-Connected : Regression
fc_regression_history = model_fc_reg.fit(x_regression_train, y_regression_train, epochs=300, batch_size=128, validation_split=.2)

#%% Plot Fully-Connected Training : Regression
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(regression_history_fc_dict['loss'],c='k', lw=1)
axes[0].plot(regression_history_fc_dict['val_loss'],'--', c='k', lw=1)
axes[0].legend(['Training','Validation'], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Loss (mean-squared error)')

axes[1].plot(regression_history_fc_dict['mae'],c='k', lw=1)
axes[1].plot(regression_history_fc_dict['val_mae'],'--',c='k', lw=1)
axes[1].legend(['Training','Validation'], loc='upper right', ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Mean absolute error')

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_regression_training.pdf')

#%% Plot Fully-Connected Training : Regression [French]
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(regression_history_fc_dict['loss'],c='k', lw=1)
axes[0].plot(regression_history_fc_dict['val_loss'],'--', c='k', lw=1)
axes[0].legend(["Entraînement","Validation"], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel("Perte (Erreur carrée moyenne)")

axes[1].plot(regression_history_fc_dict['mae'],c='k', lw=1)
axes[1].plot(regression_history_fc_dict['val_mae'],'--',c='k', lw=1)
axes[1].legend(["Entraînement","Validation"], loc='upper right', ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epoques')
axes[1].set_ylabel("Erreur moyenne absolue")

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_regression_training_fr.pdf')

#%% Test Fully-Connected : Regression

test_units_y = []
pred_units = []
poly_fit_units = []

for i in range(4):
    test_unit_x = cmapss1.get_random_test_unit(i)
    test_unit_y = cmapss1.calculate_RUL_from_dataframe(test_unit_x,'nonlinear')
    pred_unit = model_fc_reg.predict(test_unit_x.values[:,2:])
    
    x = arange(pred_unit.shape[0])
    poly_fit_unit = poly1d(polyfit(x,pred_unit.ravel(),3))(x)
    
    test_units_y.append(test_unit_y)
    pred_units.append(pred_unit)
    poly_fit_units.append(poly_fit_unit)


#%% Plot Fully-Connected : Regression
fig, axes = plt.subplots(1,2, figsize=(6,2.5), sharey=True)

axes[0].set_title('Test Unit 01')
axes[0].plot(test_units_y[0], '--', c='k', lw=1)
axes[0].plot(pred_units[0], c='k', lw=1)
axes[0].plot(poly_fit_units[0], c='r')
axes[0].grid(ls=':')
axes[0].set_xlabel('Cycles')
axes[0].set_ylabel('Remaining cycles')
axes[0].axis([0,250,0,250])

axes[1].set_title('Test Unit 02')
axes[1].plot(test_units_y[1], '--', c='k', lw=1)
axes[1].plot(pred_units[1], c='k', lw=1)
axes[1].plot(poly_fit_units[1], c='r')
axes[1].grid(ls=':')
axes[1].set_xlabel('Cycles')
#axes[1].set_ylabel('Remaining cycles')
axes[1].axis([0,210,0,250])
fig.legend(['Actual RUL','Prediction','Polynomial fit'],loc = 'lower center', ncol=3, fontsize='medium',bbox_to_anchor=(.5,.95))

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_regression_predictions.pdf',bbox_inches = 'tight')

#%% Plot Fully-Connected : Regression [French]
fig, axes = plt.subplots(1,2, figsize=(6,2.5), sharey=True)

axes[0].set_title('Unité de test 01')
axes[0].plot(test_units_y[0], '--', c='k', lw=1)
axes[0].plot(pred_units[0], c='k', lw=1)
axes[0].plot(poly_fit_units[0], c='r')
axes[0].grid(ls=':')
axes[0].set_xlabel('Cycles')
axes[0].set_ylabel('Cycles restants')
axes[0].axis([0,250,0,250])

axes[1].set_title('Unité de test 02')
axes[1].plot(test_units_y[1], '--', c='k', lw=1)
axes[1].plot(pred_units[1], c='k', lw=1)
axes[1].plot(poly_fit_units[1], c='r')
axes[1].grid(ls=':')
axes[1].set_xlabel('Cycles')
#axes[1].set_ylabel('Remaining cycles')
axes[1].axis([0,210,0,250])
fig.legend(['RUL réel','Prédiction','Ajustement polynomial'],loc = 'lower center', ncol=3, fontsize='medium',bbox_to_anchor=(.5,.95))

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_regression_predictions_fr.pdf',bbox_inches = 'tight')

#%% Prepare LSTM Data
cmapss1.sequence_length=100
cmapss1.start_of_failure = 100
lstm_train_data, lstm_test_data, lstm_train_rul, lstm_test_rul = cmapss1.get_lstm_train_test(mask_value=-10, kind='nonlinear')

#%%
cmapss2.sequence_length=100
lstm_train_data2, lstm_test_data2, lstm_train_rul2, lstm_test_rul2 = cmapss2.get_lstm_train_test(mask_value=-10, kind='nonlinear')

cmapss3.sequence_length=100
lstm_train_data3, lstm_test_data3, lstm_train_rul3, lstm_test_rul3 = cmapss3.get_lstm_train_test(mask_value=-10, kind='nonlinear')

cmapss4.sequence_length=100
lstm_train_data4, lstm_test_data4, lstm_train_rul4, lstm_test_rul4 = cmapss4.get_lstm_train_test(mask_value=-10, kind='nonlinear')

#%%
lstm_train_data_total = concatenate((lstm_train_data,lstm_train_data2,lstm_train_data3,lstm_train_data4), axis=0)
lstm_train_rul_total = concatenate((lstm_train_rul,lstm_train_rul2,lstm_train_rul3,lstm_train_rul4))
#%% LSTM : Regression
model_lstm = Sequential()
model_lstm.add( LSTM(100, return_sequences=True, input_shape=(100,24), name='lstm_1') )
model_lstm.add( LSTM(100, return_sequences=True , name='lstm_3') )
model_lstm.add( LSTM(75, name='lstm_4') )
model_lstm.add( Dense(120, activation='relu') )
model_lstm.add( Dense(110, activation='relu') )
model_lstm.add( (Dense(100)) )
model_lstm.compile( loss='mean_squared_error', metrics=['mae'], optimizer='adam')


#%% Plot LSTM Training
history = model_lstm.fit(lstm_train_data, lstm_train_rul, epochs=50, validation_split=.2, batch_size=16)


#%% Test LSTM
lstm_test_unit, lstm_test_unit_rul = cmapss1.get_lstm_test_unit(0, kind='nonlinear')

predictions_lstm = model_lstm.predict(lstm_train_data[:2,:,:])
predictions_lstm_test = model_lstm.predict(lstm_test_unit)
unit1_lstm_pred = predictions_lstm_test.ravel()[:181]
unit1_lstm_real = lstm_test_unit_rul.ravel()[:181]
lstm_test_unit, lstm_test_unit_rul = cmapss1.get_lstm_test_unit(2, kind='nonlinear')
predictions_lstm = model_lstm.predict(lstm_train_data[:2,:,:])
predictions_lstm_test = model_lstm.predict(lstm_test_unit)
unit2_lstm_pred = predictions_lstm_test.ravel()[:174]
unit2_lstm_real = lstm_test_unit_rul.ravel()[:174]

#%%
plt.plot(unit1_lstm_pred)
plt.plot(unit1_lstm_real)

#%% Plot LSTM Training
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(history.history['loss'],c='k', lw=1)
axes[0].plot(history.history['val_loss'],'--', c='k', lw=1)
axes[0].legend(['Training','Validation'], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel('Loss (mean-squared error)')

axes[1].plot(history.history['mae'],c='k', lw=1)
axes[1].plot(history.history['val_mae'],'--',c='k', lw=1)
axes[1].legend(['Training','Validation'], loc='upper right', ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Mean absolute error')

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_lstm_training.pdf')

#%% Plot LSTM Training : French
fig, axes = plt.subplots(2,1,figsize=(5.5,5), sharex=True)
axes[0].plot(history.history['loss'],c='k', lw=1)
axes[0].plot(history.history['val_loss'],'--', c='k', lw=1)
axes[0].legend(["Entraînement","Validation"], ncol=2,framealpha=.5)
axes[0].grid(ls=':')
axes[0].set_ylabel("Perte (Erreur carrée moyenne)")

axes[1].plot(history.history['mae'],c='k', lw=1)
axes[1].plot(history.history['val_mae'],'--',c='k', lw=1)
axes[1].legend(["Entraînement",'Validation'], loc='upper right', ncol=2,framealpha=.5)
axes[1].grid(ls=':')
axes[1].set_xlabel('Epoques')
axes[1].set_ylabel("Erreur moyenne absolue")

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_lstm_training_fr.pdf')

#%% Plot LSTM Test Results
fig, axes = plt.subplots(1,2, figsize=(6,2.5), sharey=True)

axes[0].set_title('Test Unit 01')
axes[0].plot(unit1_lstm_pred, c='k', lw=1)
axes[0].plot(unit1_lstm_real, '--', c='k', lw=1)

axes[0].grid(ls=':')
axes[0].set_xlabel('Cycles')
axes[0].set_ylabel('Remaining cycles')
axes[0].axis([0,200,0,250])

axes[1].set_title('Test Unit 02')
axes[1].plot(unit2_lstm_pred, c='k', lw=1)
axes[1].plot(unit2_lstm_real, '--', c='k', lw=1)
axes[1].grid(ls=':')
axes[1].set_xlabel('Cycles')
#axes[1].set_ylabel('Remaining cycles')
axes[1].axis([0,200,0,250])
fig.legend(['LSTM Prediction','Actual RUL'],loc = 'lower center', ncol=3, fontsize='medium',bbox_to_anchor=(.5,.95))

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_lstm_regression_predictions.pdf',bbox_inches = 'tight')

#%% Plot LSTM Test Results [French]
fig, axes = plt.subplots(1,2, figsize=(6,2.5), sharey=True)

axes[0].set_title('Unité de test 01')
axes[0].plot(unit1_lstm_pred, c='k', lw=1)
axes[0].plot(unit1_lstm_real, '--', c='k', lw=1)

axes[0].grid(ls=':')
axes[0].set_xlabel('Cycles')
axes[0].set_ylabel('Remaining cycles')
axes[0].axis([0,200,0,250])

axes[1].set_title('Unité de test 02')
axes[1].plot(unit2_lstm_pred, c='k', lw=1)
axes[1].plot(unit2_lstm_real, '--', c='k', lw=1)
axes[1].grid(ls=':')
axes[1].set_xlabel('Cycles')
#axes[1].set_ylabel('Remaining cycles')
axes[1].axis([0,200,0,250])
fig.legend(['Prédiction de LSTM','RUL réel'],loc = 'lower center', ncol=3, fontsize='medium',bbox_to_anchor=(.5,.95))

fig.tight_layout()
fig.savefig('D:\\Thesis\\Document\\figures\\cmapss_lstm_regression_predictions_fr.pdf',bbox_inches = 'tight')

#%% Save Training History
regression_history_fc_dict = dict(
    {
     'loss' : fc_regression_history.history['loss'],
     'val_loss' : fc_regression_history.history['val_loss'],
     'mae' : fc_regression_history.history['mae'],
     'val_mae' : fc_regression_history.history['val_mae'],
    }
    )

"""classification_history_fc_dict = dict(
    {
     'loss' : fc_classifcation_history.history['loss'],
     'val_loss' : fc_classifcation_history.history['val_loss'],
     'acc' : fc_classifcation_history.history['acc'],
     'val_acc' : fc_classifcation_history.history['val_acc'],
    } 
    )
"""
pd.DataFrame(regression_history_fc_dict).to_csv('C:/Users/abdel/Documents/Workspace/NASA/regression_history_fc.csv')
#pd.DataFrame(classification_history_fc_dict).to_csv('C:/Users/abdel/Documents/Workspace/NASA/classification_history_fc.csv')
#model_fc_clf.save('C:/Users/abdel/Documents/Workspace/NASA/model_fc_clf.h5')
model_fc_reg.save('C:/Users/abdel/Documents/Workspace/NASA/model_fc_reg.h5')
model_lstm.save('C:/Users/abdel/Documents/Workspace/NASA/model_lstm.h5')

lstm_history_dict = dict(
    {
     'loss' : history.history['loss'],
     'val_loss' : history.history['val_loss'],
     'mae' : history.history['mae'],
     'val_mae' : history.history['val_mae'],
    }
    )
pd.DataFrame(lstm_history_dict).to_csv('C:/Users/abdel/Documents/Workspace/NASA/lstm_history.csv')
