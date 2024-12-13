import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_raw = pd.read_csv('../data/user_behavior_dataset.csv')
data_transformed = pd.read_csv('../data/transformed_user_behavior_dataset.csv')

# corrplot
def corr_plot():
    corr = data_transformed.corr()
    
    sns.heatmap(corr, cmap = 'coolwarm')
    plt.show()

# density distribution plots
def density_plots():
    vars = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)']

    for i in range(len(vars)):
        sns.kdeplot(data = data_raw, 
                x = vars[i], 
                hue = 'Device Model', 
                linewidth = 0.75)
        plt.show()
