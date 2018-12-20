# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:50:55 2018

@author: Seizure
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_corr(df):
    """This function plots a graphical correlation matrix for each pair of columns in the dataframe
    The length and width of the graph are limited to the amount of columns & rows divided by 8 to increase
    ease of use of outputted plots.
    Input:
        df: pandas DataFrame"""
        
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(len(df.columns), len(df.columns)))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.imshow(df.corr(), cmap='hot', interpolation = 'nearest')
    plt.colorbar()
    fig.set_size_inches(len(df.columns)/8,len(df.columns)/8)
    plt.savefig('correlationmatrix.png', dpi = 100)


def Numerical_Data_Boxplot(DataFrame, n_vars):
    for n_var in n_vars:
        plt.figure()
        DataFrame.boxplot([n_var])
        plt.savefig('boxplot_'+str(n_var)+'.png', dpi = 100)
        plt.close()

def plot_bars(DataFrame, c_vars):
    """This function plots bar graphs for each one hot encoded variablem. The length and width
    of the graph are limited to increase ease of use of outputted plots.
    Input:
        df: pandas DataFrame"""
        
    for c_var in c_vars:
        plt.figure()
        DataFrame.plot.bar([c_var])
        plt.savefig('bargraph_'+str(c_var)+'.png', dpi = 100)
        plt.close
        