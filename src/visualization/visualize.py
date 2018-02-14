import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# confusion matrix plot
def plot_confusion_matrix(test_labels, pred):
    raw_array = metrics.confusion_matrix(test_labels, pred)

    df_cm = pd.DataFrame(raw_array, index = [0, 1],
                      columns = [0, 1])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')