'''
This module is implementations of common evaluation metrics in recommender systems
'''

import numpy as np
import heapq
import sklearn.metrics as metrics

############# Rating Prediction Metrics
def MAE(ground_truth, predictions):
    # Flatten the input array-like data into 1-D
    truth = np.asarray(ground_truth).flatten()
    pred = np.asarray(predictions).flatten()
    return metrics.mean_absolute_error(y_true=truth, y_pred=pred)

def RMS(ground_truth, predictions):
    # Flatten the input array-like data into 1-D
    truth = np.asarray(ground_truth).flatten()
    pred = np.asarray(predictions).flatten()
    return np.sqrt(metrics.mean_squared_error(y_true=truth, y_pred=pred))

# Calculate all relevant prediction metrics
def predictionMetrics(labels, predictions):
    return MAE(labels, predictions), RMS(labels, predictions)

############# Top-K Ranking Metrics
def Precision_and_Recall(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_relevant_item = 0
    for item in testinglist:
        if item in rankinglist:
            sum_relevant_item += 1

    precision = sum_relevant_item / len(rankinglist)
    recall = sum_relevant_item / len(testinglist)
    return precision, recall

def Recall(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_relevant_item = 0
    for item in testinglist:
        if item in rankinglist:
            sum_relevant_item += 1

    return sum_relevant_item / len(testinglist)

def AP(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    precision, rel = [], []
    for k in range(len(rankinglist)): # Loop the ranking list and calculate each precision for each loop

        if rankinglist[k] in testinglist: # The k-th item is relevant
            rel.append(1)
        else:
            rel.append(0)

        sum_relevant_item = 0       # Precision up-to k-th item
        rkl = rankinglist[:k+1]
        for item in testinglist:
            if item in rkl:
                sum_relevant_item += 1
        precision.append(sum_relevant_item / len(rkl))

    return np.sum(np.asarray(precision) * np.asarray(rel)) / min(len(rankinglist), len(testinglist))

def HR(ranklist, itemdict):
    rankinglist = np.asarray(ranklist, dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    assert len(testinglist) == 1 # In calculating hit rate, the length of the item list must be 1

    if testinglist[0] in rankinglist:
        return 1
    else:
        return 0

def NDCG(ranklist, itemdict):
    rankinglist = np.asarray(ranklist,dtype=np.int32).flatten()
    testinglist = np.asarray(list(itemdict.keys()),dtype=np.int32).flatten()

    dcg_sum = 0.0
    for item in testinglist:
        if item in rankinglist:
            idx = np.argwhere(rankinglist == item).item()
            dcg_sum += itemdict[item] / np.log2(idx+2) # Get the DCG sum

    idcg_sum, pos = 0.0, 0.0
    # Ideally, all items are ranked at the head of the resulting list
    ideal_ranking = sorted(itemdict.items(),key=lambda x: x[1], reverse=True)
    for item, rating in ideal_ranking:
        idcg_sum += rating / np.log2(pos+2)
        pos += 1

    return dcg_sum / idcg_sum

def MRR(ranklist, itemdict):
    rankinglist = np.asarray(ranklist).flatten()
    testinglist = np.asarray(list(itemdict.keys()), dtype=np.int32).flatten()

    sum_rr = 0.0
    for item in testinglist:
        if item in rankinglist:
            idx = np.argwhere(rankinglist == item).item()
            sum_rr += 1.0/(idx+1)

    return sum_rr / len(testinglist)

# Calculate all relevant ranking metrics
def rankingMetrics(scores, itemlist, K, test_itemdict, mod = 'precision'):
    assert len(scores) == len(itemlist)
    # Construct the score list
    scoredict = {}
    for i in range(len(scores)):
        scoredict[itemlist[i]] = scores[i]

    # Get the top K scored items
    ranklist = heapq.nlargest(K, scoredict, key=scoredict.get)

    if mod == 'hr':
        return HR(ranklist, test_itemdict), NDCG(ranklist, test_itemdict)
    if mod == 'precision':
        precision, recall = Precision_and_Recall(ranklist, test_itemdict)
        #avg_precision = AP(ranklist, test_itemdict)
        ndcg=NDCG(ranklist, test_itemdict)
        return precision, recall, ndcg

############# Top-K Ranking Metrics (Old)

def evalHR(ranklist, item):
    res = np.asarray(ranklist).flatten()
    if item in res:
        return 1
    else:
        return 0

def evalNDCG(ranklist, item):
    res = np.asarray(ranklist).flatten()

    # Caculate nDCG value
    # (Only one item is relevant in the list)
    # All the other relevance values are set to zero
    # Only the position of the target item matters
    if item in res:
        idx = np.argwhere(res == item).item()
        return 1.0 / np.log2(idx+2)
    else:
        return 0.0

def evalMRR(ranklist, item):
    res = np.asarray(ranklist).flatten()
    if item in res:
        idx = np.argwhere(res == item).item()
        return 1.0 / (idx+1)
    else:
        return 0.0

def evalTopK(scores, itemlist, K, itempos = None):
    assert len(scores) ==  len(itemlist)

    if itempos == None:
        itempos = len(itemlist)-1

    # Get the target item from the original item list
    target_item = itemlist[itempos]

    # Construct the score list
    scoredict = {}
    for i in range(len(scores)):
        scoredict[itemlist[i]] = scores[i]

    # Get the top K scored items
    ranklist = heapq.nlargest(K, scoredict, key=scoredict.get)

    # Return the metrics
    return ranklist,evalHR(ranklist,target_item),evalNDCG(ranklist,target_item),evalMRR(ranklist,target_item)

####################################################################################################################

# if __name__ == "__main__":
    # score = [4,2,3,1,7,4]
    # itemlist =[1234,5678,1111,2222,3333,4444]
    # print(evalTopK(score, itemlist, 3))