import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

def matrixSimilarity(A, B, method='pearson'):
    
    if A.shape != B.shape:
        print('Matrices must have the same dimensions')
        return
    
    nNodes = A.shape[1]
    
    if A.ndim == 2:
        if A.shape[0] == A.shape[1]:
            a = A[np.triu_indices(nNodes, k = 1)]
            b = B[np.triu_indices(nNodes, k = 1)]
        else:
            a = A.flatten()
            b = B.flatten()
        
        if method == 'cosine':
            similarity = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
        elif method == 'pearson':
            similarity = np.corrcoef((a,b))[0,1]
        elif method == 'spearman':
            similarity = stats.spearmanr(a,b)[0]
        elif method == 'jaccard':
            intersection = sum(np.where((a!=0) & (b!=0), 1, 0))
            union = sum(np.where((a!=0) | (b!=0), 1, 0))
            similarity = intersection/union
        elif method == 'weightedJaccard':
            similarity = sum(np.minimum(abs(a),abs(b)))/sum(np.maximum(abs(a),abs(b)))
        elif method == 'R2':
            similarity = r2_score(a,b)
    
    elif A.ndim == 3:
        nSubjs = A.shape[2]
        similarity = np.empty((nSubjs))
    
        for s in range(nSubjs):
            if A.shape[0] == A.shape[1]:
                a = A[:,:,s][np.triu_indices(nNodes, k = 1)]
                b = B[:,:,s][np.triu_indices(nNodes, k = 1)]
            else:
                a = A[:,:,s].flatten()
                b = B[:,:,s].flatten()
            
            if method == 'cosine':
                similarity[s] = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
            elif method == 'pearson':
                similarity[s] = np.corrcoef((a,b))[0,1]
            elif method == 'spearman':
                similarity[s] = stats.spearmanr(a,b)[0]
            elif method == 'jaccard':
                intersection = sum(np.where((a!=0) & (b!=0), 1, 0))
                union = sum(np.where((a!=0) | (b!=0), 1, 0))
                similarity[s] = intersection/union
            elif method == 'weightedJaccard':
                similarity[s] = sum(np.minimum(abs(a),abs(b)))/sum(np.maximum(abs(a),abs(b)))
            elif method == 'R2':
                similarity[s] = r2_score(a,b)
        
    return similarity