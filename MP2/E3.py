import numpy as np
from scipy.io import loadmat


# %% Confusion matrix method

def MyConfMatrix_201630945(gt, pred):
    ann = np.asarray(gt)
    pre = np.asarray(pred)
    m = np.size(ann)
    if ann.ndim != 1 or pre.ndim != 1 or m != np.size(pre):
        raise Exception('Las dimensiones de los arreglos no son correctas.')
    # It is assumed classes are enumerated from 1 to N based on the provided data.
    # The size of the matrix is therefore the maximum class number.
    n = np.max(ann)
    conf_matrix = np.zeros([n] * 2, dtype=np.int8)
    # Since matrices are indexed from 0, class i has index i-1.
    # Columns for predicted class, rows for real class (annotations).
    for i in range(m):
        conf_matrix[ann[i]-1, pred[i]-1] += 1
    # After processing the confusion matrix, calculations are performed.
    prec_class, rec_class, mean_prec, mean_rec = {}, {}, 0, 0
    for i in range(n):
        # Index i will represent the current class of interest i+1.
        # True positives is the value predicted and annotated as the class of interest.
        tp, fp, fn = conf_matrix[i, i], 0, 0
        for j in range(n):
            for k in range(n):
                if j != i:
                    if k == i:
                        # False positives are all the values predicted as the class of interest
                        # (k = i) but are annotated as another class (j != i).
                        fp += conf_matrix[j, k]
                elif k != i:
                    # False negatives are all the values predicted as not the class of interest
                    # (k != i) but are annotated as annotated as the class of interest (j = i).
                    fn += conf_matrix[j, k]
        # Adding all the total values for further average calculation.
        prec_class[i+1] = np.round(tp/(tp+fp), decimals=3)
        rec_class[i+1] = np.round(tp/(tp+fn), decimals=3)
    mean_prec = np.round(np.mean(list(prec_class.values())), decimals=3)
    mean_rec = np.round(np.mean(list(rec_class.values())), decimals=3)
    return conf_matrix, prec_class, rec_class, mean_prec, mean_rec


# %% Verification with mat file info.

data = loadmat('flower_classifier_results.mat')
results = MyConfMatrix_201630945(data['groundtruth'][0], data['predictions'][0])
print(results)
