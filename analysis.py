"""
code for draw figures
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
import scienceplots

plt.style.use('science')

if __name__ == '__main__':
    result_df = pd.read_csv('./result_df.csv')
    corr_all = result_df.corr(method='spearman')
    test_col = [col for col in list(result_df.columns) if col.endswith("test") and col.startswith("rank")]
    train_col = [col for col in list(result_df.columns) if col.endswith("val") and col.startswith("rank")]
    corr = corr_all.loc[train_col, test_col]
    corr_test = corr_all.loc[test_col, test_col]
    val_methods = ['CV', 'FCV', 'FH']

    # region corr heatmap
    with plt.style.context(['science', 'no-latex']):
        x_tick_labels = [i.split("_")[1] for i in test_col]
        y_tick_labels = [i.split("_")[1] + " " + i.split("_")[2] for i in train_col]
        x_tick_labels = [col.replace("R2", "$R^2$") if "R2" in col else col for col in x_tick_labels]
        y_tick_labels = [col.replace("R2", "$R^2$") if "R2" in col else col for col in y_tick_labels]
        plt.figure(figsize=(10, 10), dpi=300)
        mask = np.zeros_like(corr)
        with sns.axes_style("white"):
            ax = sns.heatmap(corr, annot=True, annot_kws={"size": 10}, fmt=".2f",
                             cmap="rainbow", linewidths=0, vmin=-1, vmax=1, mask=mask, square=True,
                             xticklabels=x_tick_labels, yticklabels=y_tick_labels
                             )
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            i = 0
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                i = i + 1
                tick.set_fontsize('15')
            i = 0
            for tick in ax.get_yticklabels():
                tick.set_rotation(0)
                i = i + 1
                tick.set_fontsize('15')
            fig = ax.get_figure()
        plt.xlabel("Test Metric", fontsize=20)
        plt.ylabel("Validation Metric ", fontsize=20)
        plt.savefig('./figures/1.png', bbox_inches='tight')
    # endregion

    # region corr heatmap test
    with plt.style.context(['science', 'no-latex']):
        x_tick_labels = [i.split("_")[1] for i in test_col]
        y_tick_labels = x_tick_labels
        plt.figure(figsize=(10, 10), dpi=300)
        mask = np.zeros_like(corr_test)
        with sns.axes_style("white"):
            ax = sns.heatmap(corr_test, annot=True, annot_kws={"size": 10}, fmt=".2f",
                             cmap="rainbow", linewidths=0, vmin=-1, vmax=1, mask=mask, square=True,
                             xticklabels=x_tick_labels, yticklabels=y_tick_labels
                             )

            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            i = 0
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                i = i + 1
                tick.set_fontsize('15')
            i = 0
            for tick in ax.get_yticklabels():
                tick.set_rotation(0)
                i = i + 1
                tick.set_fontsize('15')
            fig = ax.get_figure()

        plt.xlabel("Test Metric", fontsize=20)
        plt.ylabel("Test Metric ", fontsize=20)
        plt.savefig('./figures/test.png', bbox_inches='tight')
    # endregion

    # region Value consistency
    # MAE
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        metric = 'MAE'
        for method in val_methods:
            x = f'{metric} test'
            y = f'{method} {metric} val'
            plt.scatter(x=result_df[x], y=result_df[y], label=method, alpha=0.6)
        plt.xlabel(f'{metric} test')
        plt.ylabel(f'{metric} val')
        plt.xlim(0, 4.5)
        plt.ylim(0, 4.5)
        plt.plot([0, 4.5], [0, 4.5], '--', c='black')
        plt.legend(loc=2, bbox_to_anchor=(1, 1))
        plt.savefig(f'./figures/{metric}.png', bbox_inches='tight')
    # RMSE
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        metric = 'RMSE'
        for method in val_methods:
            x = f'{metric} test'
            y = f'{method} {metric} val'
            plt.scatter(x=result_df[x], y=result_df[y], label=method, alpha=0.6)
        plt.xlabel(f'{metric} test')
        plt.ylabel(f'{metric} val')
        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.plot([0, 6], [0, 6], '--', c='black')
        plt.legend(loc=2, bbox_to_anchor=(1, 1))
        plt.savefig(f'./figures/{metric}.png', bbox_inches='tight')
    # R2
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        metric = 'R2'
        for method in val_methods:
            x = f'{metric} test'
            y = f'{method} {metric} val'
            plt.scatter(x=result_df[x], y=result_df[y], label=method, alpha=0.6)
        plt.xlabel(f'$R^2$ test')
        plt.ylabel(f'$R^2$ val')
        plt.xlim(-50, 1)
        plt.ylim(-50, 1)
        plt.plot([-50, 1], [-50, 1], '--', c='black')
        plt.legend(loc=2, bbox_to_anchor=(1, 1))
        plt.savefig(f'./figures/{metric}.png', bbox_inches='tight')
    # ME
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        metric = 'ME'
        for method in val_methods[-2:]:
            x = f'{metric} test'
            y = f'{method} {metric} val'
            plt.scatter(x=result_df[x], y=result_df[y], label=method, alpha=0.6)
        plt.xlabel(f'{metric}(10) test')
        plt.ylabel(f'{metric}(10) val')
        lim_max = 8
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.legend(loc=2, bbox_to_anchor=(1, 1))
        plt.savefig(f'./figures/{metric}.png', bbox_inches='tight')
    # endregion

    # region ME10
    with plt.style.context(['science', 'no-latex']):
        x = 'ME(10) test'
        y = 'FH ME(10) val'
        result_df_filter = result_df[result_df[y] > 0]
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)

        lim_max = 10
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/ME (10)_R2.png', bbox_inches='tight')
    # endregion

    # region ME90
    with plt.style.context(['science', 'no-latex']):
        x = 'ME(90) test'
        y = 'FH ME(90) val'
        result_df_filter = result_df[result_df[y] > 0]
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        # x_data = result_df[x]
        # y_data = result_df[y]
        #
        lim_max = 12.5
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel(x, fontsize=12, fontweight='bold')
        # plt.ylabel(y, fontsize=12, fontweight='bold')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/ME (90)_R2.png', bbox_inches='tight')
    # endregion

    # region ME10 vs 90
    with plt.style.context(['science', 'no-latex']):
        x = 'ME(90) test'
        y = 'FH ME(10) val'
        result_df_filter = result_df[result_df[y] > 0]
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        #
        lim_max = 12.5
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/ME (10)v(90)_R2.png', bbox_inches='tight')
    # endregion

    # region ME30 vs 90
    with plt.style.context(['science', 'no-latex']):
        x = 'ME(90) test'
        y = 'FH ME(30) val'
        result_df_filter = result_df[result_df[y] > 0]
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)

        lim_max = 20
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/ME (30)v(90)_R2.png', bbox_inches='tight')
    # endregion

    # MAE
    with plt.style.context(['science', 'no-latex']):
        x = 'MAE test'
        y = 'FH MAE val'
        result_df_filter = result_df
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        # x_data = result_df[x]
        # y_data = result_df[y]
        #
        lim_max = 2
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel(x, fontsize=12, fontweight='bold')
        # plt.ylabel(y, fontsize=12, fontweight='bold')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/MAE_R2.png', bbox_inches='tight')
    # RMSE
    with plt.style.context(['science', 'no-latex']):
        x = 'RMSE test'
        y = 'FH RMSE val'
        result_df_filter = result_df
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        lim_max = 2
        lim_min = 0
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel(x, fontsize=12, fontweight='bold')
        # plt.ylabel(y, fontsize=12, fontweight='bold')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/RMSE_R2.png', bbox_inches='tight')
    # R2
    with plt.style.context(['science', 'no-latex']):
        x = 'R2 test'
        y = 'FH R2 val'
        result_df_filter = result_df
        r2 = r2_score(result_df_filter[x], result_df_filter[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(result_df_filter[x], result_df_filter[y], c='blue', alpha=0.6)
        x = '$R^2$ test'
        y = 'FH $R^2$ val'
        plt.xlabel(x)
        plt.ylabel(y)
        lim_max = 1
        lim_min = -100
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel(x, fontsize=12, fontweight='bold')
        # plt.ylabel(y, fontsize=12, fontweight='bold')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/R2_R2.png', bbox_inches='tight')

    # ME(x).mean in testing set
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        data_dict = {
            "x": [],
            "y": []
        }
        df2 = pd.DataFrame()
        for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            x = f'ME({i}) test'
            data_dict['x'].extend([x[:-5]] * len(result_df[x]))
            data_dict['y'].extend(list(result_df[x]))
        df_data = pd.DataFrame(data_dict)
        sns.boxplot(x="x", y="y", data=df_data, palette="Set1")
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_fontsize('8')
        plt.xlabel("")
        plt.ylabel("")
        plt.savefig(f'./figures/boxplot_ME.png', bbox_inches='tight')
