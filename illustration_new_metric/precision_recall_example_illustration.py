import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


ranking = [1,0,1,1,0,1,0,0,0,0,0,0,0]
proba_pred = np.flip(np.arange(len(ranking)))

precision, recall, _ = precision_recall_curve(ranking, proba_pred)

def precision_recall(ranking):
    ranking=np.array(ranking)
    precision = np.zeros(len(ranking)+1)
    recall = np.zeros(len(ranking)+1)
    precision[0] = 1.
    recall[0] = 0.
    for i in range(1,len(ranking)+1):
        precision[i] = ranking[:i].sum()/float(i)
        recall[i] = ranking[:i].sum()/ranking.sum()

    return precision, recall

precision, recall = precision_recall(ranking)
print(precision, recall, len(ranking))
tol = 0.1
plt.plot(recall, precision,'o-')
plt.title("Precision recall curve of the ranking")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0-tol,1+tol)
plt.ylim(0-tol,1+tol)
plt.show()