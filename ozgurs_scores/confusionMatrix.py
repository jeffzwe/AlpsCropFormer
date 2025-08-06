import pickle as pkl
import numpy as np
import pandas as pd

# Load the names
label_sheet = pd.read_csv("crop_mappings.csv")
csv_key = '4th_tier_ENG'
mapping_dict = {}
unique_codes = label_sheet[csv_key].unique()
for idx, code in enumerate(unique_codes):
    mapping_dict[code] = idx + 1
target_mapping = {0: 0, -1: 0}
for _, row in label_sheet.iterrows():
    target_mapping[int(row['LNF_code'])] = mapping_dict[row[csv_key]]
num_classes = len(unique_codes) + 1
sorted_items = sorted(mapping_dict.items(), key=lambda x: x[1])
labels = [k for k, _ in sorted_items]

labelsOI_crop1990 = ['SummerBarley', 'WinterBarley', 'Field bean', 'Fallow', 'Beets', 'Maize', 'Oat', 'Peas', 'Potatoes', 'SummerRapeseed', 'WinterRapeseed', 'Rye', 'Sugar_beets', 'Sunflowers', 'Soy', 'Spelt', 'Vegetables', 'Wheat', 'SummerWheat', 'WinterWheat']
# !! missing: Clover, Silage, Triticale
use_labelsOI = True

def load_confusion_matrix(
    path,
    labels,
    remove_firsts=False,
    normalize=True
    ):
    
    # Open a pickle file
    if path.endswith(".pkl"):
        cm = pkl.load(open(path, "rb"))

    if remove_firsts:
        cm = cm[1:,1:]

    # Get the confusion matrix as a dataframe
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Normalize the confusion matrix
    if normalize:
        cm_df = cm_df.div(cm_df.sum(axis=1), axis=0)

    return cm_df

def confusion_matrix_analysis(conf_mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        conf_mat (array or pd.DataFrame): confusion matrix

    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics

    """
    if isinstance(conf_mat, pd.DataFrame):
        names = conf_mat.index
        mat = conf_mat.to_numpy()
    else:
        mat = conf_mat
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.nansum(mat[j, j])
        fp = np.nansum(mat[:, j]) - tp
        fn = np.nansum(mat[j, :]) - tp

        d['IoU'] = tp / (tp + fp + fn + 1e-6)
        d['Precision'] = tp / (tp + fp + 1e-6)
        d['Recall'] = tp / (tp + fn + 1e-6)
        d['F1-score'] = 2 * tp / ((2 * tp + fp + fn) + 1e-6)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall['micro_IoU'] = TP / (TP + FP + FN)
    overall['micro_Precision'] = TP / (TP + FP)
    overall['micro_Recall'] = TP / (TP + FN)
    overall['micro_F1-score'] = 2 * TP / (2 * TP + FP + FN)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = macro.loc['IoU']
    overall['MACRO_Precision'] = macro.loc['Precision']
    overall['MACRO_Recall'] = macro.loc['Recall']
    overall['MACRO_F1-score'] = macro.loc['F1-score']

    overall['Accuracy'] = np.sum(np.diag(mat)) / np.sum(mat)

    per_class_df = pd.DataFrame(per_class).T
    if isinstance(conf_mat, pd.DataFrame):
        per_class_df = pd.DataFrame(per_class).T.set_index(names)

    return per_class_df, overall


# Load the confusion matrix
scenarios_o = ["d123_ours_v2_0904", "d312_ours_v2_1104"] #"d231_ours_v2_2404"
scenarios_my = ["multiyear_123_2207", "multiyear_132_2407"]

for scenario in zip(scenarios_o, scenarios_my):
    path_o = "/mnt/eo-nas1/eoa-share/projects/004_cropmaiper/utae-paps/storage/results_"+scenario[0]+"/Fold_1/conf_mat.pkl"
    path_my = "/mnt/eo-nas1/eoa-share/projects/020_crop1990/storage/results_"+scenario[1]+"/conf_mat.pkl"

    # Get the normalized confusion matrices
    cm_df_o_norm = load_confusion_matrix(path_o, labels, remove_firsts=True)
    cm_df_my_norm = load_confusion_matrix(path_my, labels, remove_firsts=True)
    diff = cm_df_my_norm - cm_df_o_norm

    # Get the absolute confusion matrices
    cm_df_o = load_confusion_matrix(path_o, labels, remove_firsts=True, normalize=False).fillna(0)
    cm_df_my = load_confusion_matrix(path_my, labels, remove_firsts=True, normalize=False).fillna(0)
    cm_o_metrics, _ = confusion_matrix_analysis(cm_df_o)
    cm_my_metrics, _ = confusion_matrix_analysis(cm_df_my)
    freq_df = pd.read_csv(
        "/mnt/eo-nas1/eoa-share/projects/004_cropmaiper/utae-paps/lnf_code_2021_mapped_pixel_counts.txt",
        skiprows=1,
        header=None,
        names=['crop_name', 'count'],
        skipinitialspace=True
    )
    freq_df['crop_name'] = freq_df['crop_name'].replace({
        'SugarBeets': 'Sugar_beets',
        'Non-agriculture': 'Non agriculture'
        })
    freq_df = pd.concat([
        freq_df,
        pd.DataFrame([{'crop_name': 'Hemp', 'count': 0}])
    ], ignore_index=True)
    df_counts_filtered = freq_df[freq_df['crop_name'].isin(cm_my_metrics.index)].set_index('crop_name')
    metrics_o = cm_o_metrics.merge(df_counts_filtered, left_index=True, right_index=True).sort_values("count", ascending=False)
    metrics_my = cm_my_metrics.merge(df_counts_filtered, left_index=True, right_index=True).sort_values("count", ascending=False)
    metrics_diff = metrics_my.drop(columns='count') - metrics_o.drop(columns='count')
    metrics_diff['count'] = metrics_my['count']

    if use_labelsOI:
        diff = diff.loc[labelsOI_crop1990, labelsOI_crop1990]
        cm_df_my_norm = cm_df_my_norm.loc[labelsOI_crop1990, labelsOI_crop1990]
        metrics_my = metrics_my.loc[labelsOI_crop1990].sort_values("count", ascending=False)
        metrics_diff = metrics_diff.loc[labelsOI_crop1990].sort_values("count", ascending=False)

    ### Save metrics and confusion matrices
    if use_labelsOI:
        cm_df_my_norm.to_excel("storage/results_"+scenario[1]+"/confusion_matrix_normalized_labelOI.xlsx", index=True)
        cm_df_my_norm.to_excel("storage/results_"+scenario[1]+"/confusion_matrix_normalized_labelOI_diffToSingleYear.xlsx", index=True)
        metrics_my.to_excel("storage/results_"+scenario[1]+"/performance_metrics_labelOI.xlsx", index=True)
        metrics_diff.to_excel("storage/results_"+scenario[1]+"/performance_metrics_labelOI_diffToSingleYear.xlsx", index=True)
    else:
        cm_df_my_norm.to_excel("storage/results_"+scenario[1]+"/confusion_matrix_normalized.xlsx", index=True)
        cm_df_my_norm.to_excel("storage/results_"+scenario[1]+"/confusion_matrix_normalized_diffToSingleYear.xlsx", index=True)
        metrics_my.to_excel("storage/results_"+scenario[1]+"/performance_metrics.xlsx", index=True)
        metrics_diff.to_excel("storage/results_"+scenario[1]+"/performance_metrics_diffToSingleYear.xlsx", index=True)

    ### PLOT DIFFERENCE
    import matplotlib.pyplot as plt
    import seaborn as sns
    # increase figure size for better readability
    plt.figure(figsize=(16, 14))
    # draw heatmap
    ax = sns.heatmap(diff, 
                cmap="PiYG",
                square=True,
                linewidths=2,
                vmin=-0.1,
                vmax=0.1,
                xticklabels=True, 
                yticklabels=True, 
                cbar=False
                )
    # Colorbar
    heatmap = ax.get_children()[0]
    cbar = plt.colorbar(heatmap,
                        ax=ax,
                        fraction=0.035,
                        shrink=0.4,
                        pad=0.02)
    cbar.set_label("Fraction", fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    # label axes
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.title("Confusion Matrix", fontsize=20)
    # rotate x-axis labels for readability
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    if use_labelsOI:
        plt.savefig("storage/results_"+scenario[1]+"/confusionMatrix_labelOI_diffToSingleYear.pdf")
    else:
        plt.savefig("storage/results_"+scenario[1]+"/confusionMatrix_diffToSingleYear.pdf")
    # plt.show()
    plt.close()


    ### PLOT ACTUAL
    import matplotlib.pyplot as plt
    import seaborn as sns
    # increase figure size for better readability
    plt.figure(figsize=(16, 14))
    # draw heatmap
    ax = sns.heatmap(cm_df_my_norm, 
                cmap="viridis",
                square=True,
                linewidths=2,
                vmin=0,
                vmax=1,
                xticklabels=True, 
                yticklabels=True, 
                cbar=False
                )
    # Colorbar
    heatmap = ax.get_children()[0]
    cbar = plt.colorbar(heatmap,
                        ax=ax,
                        fraction=0.035,
                        shrink=0.4,
                        pad=0.02)
    cbar.set_label("Fraction", fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    # label axes
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.title("Confusion Matrix", fontsize=20)
    # rotate x-axis labels for readability
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    if use_labelsOI:
        plt.savefig("storage/results_"+scenario[1]+"/confusionMatrix_labelOI.pdf")
    else:
        plt.savefig("storage/results_"+scenario[1]+"/confusionMatrix.pdf")
    # plt.show()
    plt.close()


