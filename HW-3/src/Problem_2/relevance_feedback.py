import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def calc(vec_docs,rel,alpha, neg):
    if neg == True : 
        return alpha*np.sum(vec_docs[~np.array(rel), :], axis = 0)
    if neg == False: 
        return alpha*np.sum(vec_docs[np.array(rel), :], axis = 0)

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    
    nq = np.shape(vec_queries)[0]
    alpha = 0.75
    beta = 0.15
    for j in range(3):
        for i in range(nq):
            rel = np.argsort(-sim[:, i])[:n]
            a = calc(vec_docs,rel,alpha,neg= False) - calc(vec_docs,rel,beta,neg=True)
            vec_queries[i,:] += a
        sim = cosine_similarity(vec_docs, vec_queries)       
    return sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    
    names = tfidf_model.get_feature_names()
    nq = np.shape(vec_queries)[0]
    alpha = 0.8
    beta = 0.2
    

    for j in range(3):
        s  = ""
        for i in range(nq):
            rel = np.argsort(-sim[:, i])[:n]
            a = calc(vec_docs,rel,alpha,neg= False) - calc(vec_docs,rel,beta,neg=True)
            vec_queries[i,:] += a
            l = []
            for row in rel:
                s= ""
                ele = vec_docs.getrow(row).toarray()[0].ravel()
                top = np.argsort(ele)[-10:]
                for k in top:
                    print(k)
                    s += names[k] + ' ';
                l.append(s)
                print(s)
            vec_queries[i, :] += np.sum(tfidf_model.transform(l), axis=0)
        sim = cosine_similarity(vec_docs, vec_queries) 
    return sim