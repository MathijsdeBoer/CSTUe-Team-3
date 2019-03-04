import numpy as np
import copy

def dice_score(ground_truth, mask, prediction):
    """ Calculate dice score"""

    true_positive_mask = np.logical_and(ground_truth==1, prediction==1)
    false_positive_mask = np.logical_and(ground_truth==0, prediction==1)
    false_negative_mask = np.logical_and(ground_truth==1, prediction==0)

    TP = np.count_nonzero(np.logical_and(true_positive_mask, mask==1))
    FP = np.count_nonzero(np.logical_and(false_positive_mask, mask==1))
    FN = np.count_nonzero(np.logical_and(false_negative_mask, mask==1))

    DSC = 2*TP / (2*TP + FP + FN)
    return DSC

def optimal_treshold(ground_truth, mask, prediction, low=0.2, high=0.95, steps=100):
    """ Find optimal probability threshold that maximizes dice score for a given
    segmentation prediction. Assumes that prediction is padded by 16."""

    # Remove the padding from prediction
    prediction = prediction[16:-16,16:-16]
    # Normalize
    prediction /= np.amax(prediction)
    ground_truth = np.float32(ground_truth)
    ground_truth /= np.amax(ground_truth)
    mask = np.float32(mask)
    mask /= np.amax(mask)
    # Range of threshold values to be tested
    limits = np.linspace(low, high, steps)

    dice_scores = []
    for limit in limits:
        thres_seg = copy.deepcopy(prediction)
        thres_seg[thres_seg>limit] = 1
        thres_seg[thres_seg<=limit] = 0
        d = dice_score(ground_truth, mask, thres_seg)
        dice_scores.append(d)

    # Find the optimal threshold (maximum dice score)
    opt_pos = np.argmax(dice_scores)
    opt_threshold = limits[opt_pos]
    opt_seg = copy.deepcopy(prediction)
    opt_seg[opt_seg>opt_threshold] = 1
    opt_seg[opt_seg<=opt_threshold] = 0
    opt_dice = dice_scores[opt_pos]

    return opt_seg, opt_threshold, opt_dice
