import os
import numpy as np
import pandas as pd

# This code performs a hill climbing algorithm to create an ensemble of base models by optimizing the ensemble's AUC (Area Under the ROC Curve) score. Here's a step-by-step explanation of the code:

### 1. Initialization and Data Preparation
PATH = '../input/melanoma-oof-and-sub/'
FILES = os.listdir(PATH)
OOF = np.sort([f for f in FILES if 'oof' in f])
OOF_CSV = [pd.read_csv(PATH + k) for k in OOF]

# - `PATH`: Directory containing the out-of-fold (OOF) predictions of the models.
# - `FILES`: List of all files in the directory.
# - `OOF`: Sorted list of OOF prediction files.
# - `OOF_CSV`: List of DataFrames, each containing OOF predictions.

### 2. Extract Predictions and True Labels

x = np.zeros((len(OOF_CSV[0]), len(OOF)))
for k in range(len(OOF)):
    x[:, k] = OOF_CSV[k].pred.values

TRUE = OOF_CSV[0].target.values

# - `x`: A matrix where each column represents the predictions of a model.
# - `TRUE`: True target values (assumed to be the same for all OOF files).

### 3. Evaluate Initial AUC for Each Model
from sklearn.metrics import roc_auc_score
all = []
for k in range(x.shape[1]):
    auc = roc_auc_score(OOF_CSV[0].target, x[:,k])
    all.append(auc)
    print('Model %i has OOF AUC = %.4f' % (k, auc))

# - `all`: List to store AUC scores for each model.
# - Loop through each model's predictions to compute and print the AUC score.

### 4. Initialize the Ensemble with the Best Single Model

m = [np.argmax(all)]
w = []

old = np.max(all)

# - `m`: List to store indices of models included in the ensemble (starting with the best single model).
# - `w`: List to store weights for the models in the ensemble.
# - `old`: Best AUC score so far.

### 5. Define Search Parameters
RES = 200
PATIENCE = 10
TOL = 0.0003
DUPLICATES = False

# - `RES`: Resolution for weight search.
# - `PATIENCE`: Patience for early stopping in weight search.
# - `TOL`: Minimum improvement required to continue adding models.
# - `DUPLICATES`: Whether to allow duplicate models in the ensemble.

### 6. Hill Climbing to Optimize Ensemble

print('Ensemble AUC = %.4f by beginning with model %i' % (old, m[0]))
print()

for kk in range(len(OOF)):
    # Build current ensemble prediction
    # m[0] is the index of the initial best model.
    # x[:, m[0]] extracts the predictions of this model.
    md = x[:, m[0]]
    for i, k in enumerate(m[1:]):
        # Update the ensemble predictions by combining the current ensemble md with the predictions 
        # of the model x[:, k] using weight w[i].
        md = w[i] * x[:, k] + (1 - w[i]) * md

    # Search for the best model to add    
    # mx: Stores the maximum AUC score found in this iteration.
    mx = 0 
    # mx_k: Stores the index of the model that gives the best AUC score when added.
    mx_k = 0
    # mx_w: Stores the weight of the best model when added to the ensemble.
    mx_w = 0
    print('Searching for best model to add... ')

    # The for loop iterates over all models (x.shape[1] is the number of models).
    for k in range(x.shape[1]):
        # Print the current model index k.
        print(k, ', ', end='')
        # Skip models that are already in the ensemble 
        if not DUPLICATES and (k in m):
            continue

        # Evaluate adding model k with different weights
        # Initialize variables for tracking the best weight for the current model k
        # Best weight found for model k.
        bst_j = 0
        # Best AUC score achieved with model k.
        bst = 0
        # Counter to track the number of non-improving iterations for early stopping.
        ct = 0
        # Inner for loop iterates over possible weights (RES determines the resolution of weights from 0 to 1):
        # RES = 200, so we iterate over 0, 0.005, 0.01, ..., 0.995, 1
        for j in range(RES):
            tmp = j / RES * x[:, k] + (1 - j / RES) * md
            #  Calculate the AUC score of the ensemble with the current weight.
            auc = roc_auc_score(TRUE, tmp)
            # If the AUC score auc is better than the best score bst found so far:
            if auc > bst:
                # Update bst with the new best score auc.
                bst = auc
                # Update bst_j with the current weight j / RES.
                bst_j = j / RES
            else:
                # If no improvement is found, increment the counter ct.
                ct += 1
            # If the counter ct is greater than PATIENCE, break out of the inner loop.                
            if ct > PATIENCE:
                break
        # If the best AUC score bst for the current model k is better than the maximum AUC score mx found so far:
        # Update mx to bst.
        # Update mx_k to k.
        # Update mx_w to bst_j.                
        if bst > mx:
            mx = bst
            mx_k = k
            mx_w = bst_j

    # Check if improvement is significant
    # Calculate the improvement inc as the difference between the new best AUC score mx and the previous best score old.
    inc = mx - old
    # If the improvement inc is less than or equal to the tolerance TOL:
    # Print a message indicating no significant increase.
    # Break the loop, stopping the search for additional models.
    if inc <= TOL:
        print()
        print('No increase. Stopping.')
        break

    # Update ensemble with the new model and weight
    print()
    print('Ensemble AUC = %.4f after adding model %i with weight %.3f. Increase of %.4f' % (mx, mx_k, mx_w, inc))
    print()

    old = mx
    m.append(mx_k)
    w.append(mx_w)

# - For each iteration (up to the number of models):
#     - **Build Current Ensemble**: Start with the best initial model and iteratively combine it with other models in `m` using their weights `w`.
#     - **Search for Best Model to Add**: For each model not already in the ensemble, evaluate it with different weights to find the best one to add.
#     - **Evaluate Each Weight**: For each candidate model, iterate over possible weights to find the weight that gives the highest AUC. Use early stopping (`PATIENCE`) to terminate weight search early if no improvement is seen.
#     - **Check for Improvement**: If the best AUC improvement (`inc`) is less than `TOL`, stop the process.
#     - **Update Ensemble**: If improvement is sufficient, update the ensemble with the new model and weight, then print the updated AUC and improvement.

### Summary

# This code performs a hill climbing optimization to build an ensemble of models, starting with the best single model 
# and iteratively adding models that improve the ensemble's AUC score. The process continues until no significant improvement 
# is found. This method helps to create a more robust predictive model by combining the strengths of multiple models.