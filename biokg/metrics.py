import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)

# Taken from Pat Walter's code that I now can't find
def calc_classification_metrics(df_in, cycle_col="cv_cycle",
                                         val_col="y",
                                         prob_col="y_prob",
                                         pred_col="y_pred"):
    """
    Calculate classification metrics (ROC AUC, PR AUC, MCC)
    :param df_in: input dataframe must contain columns [method, split] as well the columns specified in the arguments
    :param cycle_col: column indicating the cross-validation fold
    :param val_col: column with the group truth value
    :param prob_col: column with probability (e.g. from sklearn predict_proba)
    :param pred_col: column with binary predictions (e.g. from sklearn predict)
    :return: a dataframe with [cv_cycle, method, split, roc_auc, pr_auc, mcc]
    """
    metric_list = []
    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k

        y_true = v[val_col].tolist()
        y_score=v[prob_col].tolist()
        y_pred = v[pred_col].tolist()        
        
        roc_auc =  roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        mcc = matthews_corrcoef(y_true, y_pred)
        metric_list.append([cycle, method, split, roc_auc, pr_auc, mcc])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split", "roc_auc", "pr_auc", "mcc"])
    return metric_df

def make_boxplots(df):
    """
    Plot box plots showing comparisons of [roc_auc, pr_auc, mcc], p-value for Friedman's test is shown as the plot title
    :param df: input dataframe, must contain [cv_cycle, method, roc_auc, pr_auc, mcc]
    """
    sns.set_context('notebook')
    sns.set_theme(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    sns.set_style('whitegrid')
    figure, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 8))

    for i, stat in enumerate(["roc_auc", "pr_auc", "mcc"]):
        friedman = pg.friedman(df, dv=stat, within="method", subject="cv_cycle")['p-unc'].values[0]
        ax = sns.boxplot(x=stat, y="method", ax=axes[i], data=df)
        title = stat.replace("_", " ").upper()
        ax.set_title(f"p={friedman:.03f}")
        ax.set_ylabel("")
        ax.set_xlabel(title)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    
    