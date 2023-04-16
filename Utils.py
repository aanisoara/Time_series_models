{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

# Module du projet de série temp : réalisé par Anisoara, Eunice et Gaoussou

#----------------------------------------------------------PROJET SERIE TEMP------------------------------------------------------------ 




#Les Librairies utilisées 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import matplotlib.pyplot
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import Utils as utils

#--------------------------------------------Fonction pour tracer les graphiques des séries----------------------------------------------

def plot_time_series(df, column_plot):
    column_plot = ['VIX', 'Parkinson', 'Squared returns']
    """
    cette fonction crée des graphiques de séries temporelles pour chaque colonne spécifiée dans la liste column_plot.
    
    df : la base de donnée contenant des colonnes à tracer 
    column_plot = Liste des colonnes 
    """
    axes = df[column_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
    for ax in axes:
        ax.set_ylabel('Valeurs')
        
#------------------------------------------Fonction pour calculer le moment des séries----------------------------------------------------

def compute_stats(df):
    # Calculer la moyenne de chaque colonne
    means = df.mean()

    # Calculer la variance de chaque colonne
    variances = df.var()

    # Calculer l'écart-type de chaque colonne
    stds = df.std()

    # Afficher les résultats pour chaque colonne
    for col in df.columns:
        print(15*' - ', col, 15*' - ')
        print("Moyenne de la série {} : {}".format(col, means[col]))
        print("Variance de la série {} : {}".format(col, variances[col]))
        print("Ecart type de la série {} : {}".format(col, stds[col]))
        print('     ')
        
#__________________________________________________________QUESTION 1_____________________________________________________________________


#---------------------------------------- Fonction qui permet de normaliser les séries---------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df


#------------------------------------Visualisation des données normaliser----------------------------------------------------------------- 
import matplotlib.pyplot as plt

def plot_macro_variables(df):
    """
    Affiche l'évolution des différentes variables macros dans le DataFrame df.
    """
    for col in df.columns:
        plt.figure(figsize = (16, 2))
        df[col].plot()
        plt.title(str(col))
        plt.show()
#________________________________________________QUESTION 2______________________________________________________________________________



#--------------------------------Test de stationnarité de nos séries-----------------------------------------------------------------------

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

def visualize_adfuller_results(series, title,df, ax):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if (p_val < significance_level) & ((adf_stat < crit_val_1)):
        linecolor = 'forestgreen' 
    elif (p_val < significance_level) & (adf_stat < crit_val_5):
        linecolor = 'orange'
    elif (p_val < significance_level) & (adf_stat < crit_val_10):
        linecolor = 'red'
    else:
        linecolor = 'purple'
    sns.lineplot(x=df.index, y=series, ax=ax, color=linecolor) 
    ax.set_title(f'ADF Statistic {adf_stat:0.3f}, p-value: {p_val:0.3f}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}', fontsize=14)
    ax.set_ylabel(ylabel=title, fontsize=14)
    
    
def visualize_series(df):
    f, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 9))

    series_names = ['VIX', 'Parkinson', 'Squared returns']
    #titles = ['Test de stationnarité de la série VIX',
    #          'Test de stationnarité de la série Parkinson',
    #          'Test de stationnarité de la série Squared returns']

    for i in range(len(series_names)):
        series_name = series_names[i]
        #title = titles[i]

        #print(15*' - ', title, 15*' - ')
        #print('     ')

        visualize_adfuller_results(df[series_name].values, series_name, df, ax[i])

    plt.tight_layout()
    plt.show()
    
    
#----------------------------------------------------Complément du test de stationnarité de nos séries-------------------------------------
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_autocorrelation(dataframe, columns, lags=100, figsize=(20, 16)):
    f, ax = plt.subplots(nrows=len(columns), ncols=1, figsize=figsize)

    for i, col in enumerate(columns):
        plot_acf(dataframe[col], lags=lags, ax=ax[i])
        ax[i].set_title(f'Autocorrelation de la série {col}')
    
    plt.show()

    
    
#------------------------------------------Fonction pour déterminer l'ordre du modèle AR-------------------------------------------------- 

import statsmodels.api as sm

def select_order_ar(time_series, max_p=20):
    aic_values = {}
    bic_values = {}
    for p in range(max_p+1):
        ar_model = sm.tsa.AutoReg(time_series, lags=p, trend='c')
        ar_result = ar_model.fit()
        aic_values[p] = ar_result.aic # Calcul de l'AIC pour le modèle AR(p)
        bic_values[p] = ar_result.bic # Calcul du BIC pour le modèle AR(p)

    # Sélection de l'ordre optimal en utilisant les critères d'information AIC et BIC
    optimal_order = min(aic_values, key=aic_values.get) if min(aic_values.values()) < min(bic_values.values()) else min(bic_values, key=bic_values.get)
    return optimal_order


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

def plot_pacf_series(df):
    f, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 16))

    plot_pacf(df['VIX'], lags=20, ax=ax[0])
    ax[0].set_title('Autocorrelation Partielle de la série VIX')

    plot_pacf(df['Parkinson'], lags=20, ax=ax[1])
    ax[1].set_title('Autocorrelation Partielle de la série Parkinson')

    plot_pacf(df['Squared returns'], lags=20, ax=ax[2])
    ax[2].set_title('Autocorrelation Partielle de la série Squared returns')

    plt.show()
    
#--------------------------------------------------------Estimation du modèle AR-----------------------------------------------------------
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

def fit_ar_models(df):
    # Estimate AR model on VIX
    model_vix = AutoReg(df['VIX'], lags=10, trend='c', old_names=False)
    results_vix = model_vix.fit()
    print('AR Model for VIX:')
    print(results_vix.summary())
    
    # Estimate AR model on Parkinson
    model_parkinson = AutoReg(df['Parkinson'], lags=10, trend='c', old_names=False)
    results_parkinson = model_parkinson.fit()
    print('AR Model for Parkinson:')
    print(results_parkinson.summary())
    
    # Estimate AR model on Squared Returns
    model_squared = AutoReg(df['Squared returns'], lags=10, trend='c', old_names=False)
    results_squared = model_squared.fit()
    print('AR Model for Squared Returns:')
    print(results_squared.summary())
    
    return results_vix, results_parkinson, results_squared
    
#______________________________________________QUESTION 3 : Visualisation des valeurs ajustés____________________________________________

def plot_ar_fitted_values(df, results_vix, results_parkinson, results_squared):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['VIX'], label='Original')
    ax.plot(results_vix.fittedvalues, label='Fitted')
    ax.set_title('AR Model Fitted Values for VIX')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['Parkinson'], label='Original')
    ax.plot(results_parkinson.fittedvalues, label='Fitted')
    ax.set_title('AR Model Fitted Values for Parkinson')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['Squared returns'], label='Original')
    ax.plot(results_squared.fittedvalues, label='Fitted')
    ax.set_title('AR Model Fitted Values for Squared Returns')
    ax.legend()
    plt.show()
    
    
    
#_____________________________________________________________QUESTION 4 : Estimation du modèle HAR_______________________________________
from arch.univariate import HARX

def fit_har_models(df):
    # Estimation du modèle HAR pour la colonne B (VIX)
    har_VIX = HARX(df['VIX'], lags=[1, 5, 22], constant=True)
    res_VIX = har_VIX.fit()
    print(res_VIX.summary())
    
    # Estimation du modèle HAR pour la colonne C (parkinson)
    har_Parkinson = HARX(df['Parkinson'], lags=[1, 5, 22], constant=True)
    res_Parkinson = har_Parkinson.fit()
    print(res_Parkinson.summary())
    
    # Estimation du modèle HAR pour la colonne D (squared returns)
    har_Squared_returns = HARX(df['Squared returns'], lags=[1, 5, 22], constant=True)
    res_Squared_returns = har_Squared_returns.fit()
    print(res_Squared_returns.summary())
    
    return res_VIX, res_Parkinson, res_Squared_returns


#-----------------------------------------------------Fonction pour comparer les deux modèles :-------------------------------------------

from arch.univariate import HARX
from statsmodels.tsa.ar_model import AutoReg

def compare_ar_har_models(df):
    # Estimate AR model on VIX
    model_vix = AutoReg(df['VIX'], lags=5, trend='c', old_names=False)
    results_vix = model_vix.fit()
    
    # Estimate AR model on Parkinson
    model_parkinson = AutoReg(df['Parkinson'], lags=14, trend='c', old_names=False)
    results_parkinson = model_parkinson.fit()
    
    # Estimate AR model on Squared Returns
    model_squared = AutoReg(df['Squared returns'], lags=16, trend='c', old_names=False)
    results_squared = model_squared.fit()
    
    # Estimation du modèle HAR pour la colonne B (VIX)
    har_VIX = HARX(df['VIX'], lags=[1, 5, 22], constant=True)
    res_VIX = har_VIX.fit()
    
    # Estimation du modèle HAR pour la colonne C (parkinson)
    har_Parkinson = HARX(df['Parkinson'], lags=[1, 5, 22], constant=True)
    res_Parkinson = har_Parkinson.fit()
    
    # Estimation du modèle HAR pour la colonne D (squared returns)
    har_Squared_returns = HARX(df['Squared returns'], lags=[1, 5, 22], constant=True)
    res_Squared_returns = har_Squared_returns.fit()
    
    print('\nLog-Likelihoods:')
    print(15*' - ', 'VIX', 15*' - ')
    print('     ')
    print('AR model for VIX: ', results_vix.llf)
    print('HAR model for VIX: ', res_VIX.loglikelihood)

    if res_VIX.loglikelihood >= results_vix.llf:
        print('HAR model is preferred for VIX')
        vix_fitted_values = res_VIX.fittedvalues
    else:
        print('AR model is preferred for VIX')
        vix_fitted_values = results_vix.fittedvalues


    print(15*' - ', 'Parkinson', 15*' - ')
    print('     ')

    print('AR model for Parkinson: ', results_parkinson.llf)
    print('HAR model for Parkinson: ', res_Parkinson.loglikelihood)

    if res_Parkinson.loglikelihood >= results_parkinson.llf:
        print('HAR model is preferred for Parkinson')
        parkinson_fitted_values = res_Parkinson.fittedvalues
    else:
        print('AR model is preferred for Parkinson')
        parkinson_fitted_values = results_parkinson.fittedvalues


    print(15*' - ', 'Squared Returns', 15*' - ')
    print('     ')

    print('AR model for Squared Returns: ', results_squared.llf)
    print('HAR model for Squared Returns: ', res_Squared_returns.loglikelihood)

    if res_Squared_returns.loglikelihood >= results_squared.llf:
        print('HAR model is preferred for Squared_returns')
        squared_fitted_values = res_Squared_returns.fittedvalues
    else:
        print('AR model is preferred for Squared_returns')
        squared_fitted_values = results_squared.fittedvalues
        
    return vix_fitted_values, parkinson_fitted_values, squared_fitted_values

#_________________________________________QUESTION 5 : Test de stationarité de nos séries estimées_______________________________________

from statsmodels.tsa.stattools import adfuller

def adf_test(data, name):
    print(15*' - ','Test de stationnarité de la série '+ name, 15*' - ')
    print('     ')

    result = adfuller(data)
    print(f'ADF Statistic for {name}:', result[0])
    print(f'p-value for {name}:', result[1])
    print(f'Critical Values for {name}:')
    for key, value in result[4].items():
        print('\t', key, ": ", value)


#___________________________________________QUESTION 6 : Estimation du modèle VAR(P)______________________________________________________

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

def estimate_var_model(vix_fitted_values, parkinson_fitted_values, squared_fitted_values, p=2):
    # Combine the fitted values into a single DataFrame
    fitted_values = pd.concat([vix_fitted_values, parkinson_fitted_values, squared_fitted_values], axis=1)
    fitted_values.columns = ["VIX", "Parkinson", "Squared Returns"]
    fitted_values = fitted_values.dropna()
    
    # Estimate the VAR model
    model = VAR(fitted_values)
    results = model.fit(p)
    
    # Print the results
    print(results.summary())
    
    return results


#____________________________________________________Question 7 : Fonction de réponse____________________________________________________

import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.irf import IRAnalysis

def plot_impulse_response(results: IRAnalysis, maxlags: int = 10) -> None:
    """
    Traces the impulse response function obtained from a VAR model.
    
    Parameters:
        - results: The result object obtained after fitting a VAR model.
        - maxlags: Maximum number of lags to use in the calculation of the impulse response function.
                   Default is 10.
    """
    irf = results.irf(maxlags)
    fig = irf.plot()
    plt.tight_layout()
    plt.show()









# ------------------------------------

def is_stationary_first_order(df, cols):
    """
    bool: True if all specified columns are stationary of first order, False otherwise.
    Pour identification de l'ordre de processus, nous nous avons appuyer sue le fait qu'un processus stochastique est considéré comme stationnaire de premier ordre si sa moyenne et sa variance sont constantes et que sa fonction d'autocorrélation dépend uniquement de l'interval de temps entre deux observations et non de l'emplacement dans le temps.
    """
    for col in cols:
        # Calculate the mean and variance over the entire dataset
        mean = np.mean(df[col])
        var = np.var(df[col])

        # Split the data into two halves
        half = len(df) // 2
        first_half = df[col].iloc[:half]
        second_half = df[col].iloc[half:]

        # Calculate the mean and variance of each half
        mean1 = np.mean(first_half)
        mean2 = np.mean(second_half)
        var1 = np.var(first_half)
        var2 = np.var(second_half)

        # Check if the mean and variance are constant
        if not np.isclose(mean1, mean2) or not np.isclose(var1, var2):
            return False

        # Calculate the autocorrelation function
        corr = []
        for i in range(1, len(df)//2):
            corr.append(np.corrcoef(df[col][:-i], df[col][i:])[0, 1])
        corr = np.array(corr)

        # Check if the autocorrelation function depends only on the time interval
        if not np.allclose(corr, corr[0]):
            return False

    return True


if is_stationary_first_order(df[['VIX', 'Parkinson', 'Squared returns']], ['VIX', 'Parkinson', 'Squared returns']):
    print("The specified columns are stationary of first order.")
else:
    print("The specified columns are not stationary of first order.")

"""
### Verifier si le processus est stationnaire de second ordre sous deux conditions : 
- vérifier si la moyenne est constante: 
La moyenne est constante dans le temps : 
E[X(t)] = μ pour tout t

Pour vérifier si la moyenne est constante dans le temps. Pour ce faire, il faut calculer la moyenne du processus à différents instants et vérifier si elle reste constante. Si la moyenne est constante, alors la première propriété est vérifiée.

- Calculer la fonction d'autocorrélation. 
La fonction d'autocorrélation dépend uniquement de la différence de temps entre deux instants : 
R(t1, t2) = R(τ), où τ = t1 - t2.

Pour ce faire, il faut choissir deux instants t1 et t2; calculer la covariance entre X(t1) et X(t2) et la diviser par la variance de X(t). il faut egalement répéter cette opération pour différents intervalles de temps entre t1 et t2 et vérifier si la fonction d'autocorrélation ne dépend que de la différence de temps. Si la fonction d'autocorrélation ne dépend que de la différence de temps, alors la deuxième propriété est vérifiée.

Si les deux propriétés sont vérifiées, alors le processus est stationnaire de second ordre. Pour se faire, nous avons creer une fonction check_stationarity qui nous permet de repondre à cette question:
"""






def check_stationarity(df, cols):
    for col in cols:
        # Récupérer les données de la colonne
        data = df[col].values

        # Calculer la moyenne
        mean = np.mean(data)

        # Calculer la fonction d'autocorrélation
        corr = np.correlate(data - mean, data - mean, mode='full')
        corr = corr[len(corr)//2:]
        lags = np.arange(len(corr))

        # Tracer la fonction d'autocorrélation
        plt.plot(lags, corr)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title(f'Autocorrelation for {col}')
        plt.show()

        # Vérifier si les propriétés sont vérifiées
        is_stationary = np.allclose(corr, corr[::-1])
        if is_stationary:
            print(f"{col} est un processus stationnaire de second ordre")
        else:
            print(f"{col} n'est pas un processus stationnaire de second ordre")

check_stationarity(df, ['VIX', 'Parkinson', 'Squared returns'])