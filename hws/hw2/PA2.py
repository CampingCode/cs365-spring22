import pandas as pd
import numpy as np
import re

def load(path):
    df = pd.read_csv('TRAIN_balanced_ham_spam.csv')

    return df
def prior(df):
    ham_prior = 0
    spam_prior =  0
    '''YOUR CODE HERE'''
    for i in df['label_num']:
        if i == 1:
            spam_prior += 1
        else:
            ham_prior += 1
            
    total = ham_prior+ spam_prior
    
    spam_prior = spam_prior / total
    ham_prior = ham_prior/ total 

    '''END'''
    return ham_prior, spam_prior

def likelihood(df):
    ham_like_dict = {}
    spam_like_dict = {}
    '''YOUR CODE HERE'''
    
    templist = []
    templist2 = []
    
    df_list = df[['text','label_num']]
    
    spamset = set()
    hamset = set()
    
    ham = df_list.loc[df['label_num'] == 0]
    spam = df_list.loc[df['label_num'] == 1]
 
    for email in spam.index:
        templist = spam['text'][email].split()
    for word in templist:
        if word not in spamset:
            spamset.add(word)
    for word in spam_set:
        if word not in spam_like_dict:
            spam_like_dict[word] = 1
        else:
            spam_like_dict[word] +=1
        
    spamset.clear()
    

    for email in ham.index:
        templist = ham['text'][email].split()
    for word in templist2:
        if word not in hamset:
            hamset.add(word)
    for word in ham_set:
        if word not in ham_like_dict:
            ham_like_dict[word] = 1
        else:
            ham_like_dict[word] +=1
        
    hamset.clear()
    
    
    
    '''END'''

    return ham_like_dict, spam_like_dict


def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
    '''
    prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
    '''
    #ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
    ham_spam_decision = None




    '''YOUR CODE HERE'''
    #ham_posterior = posterior probability that the email is normal/ham
    ham_posterior = None

    #spam_posterior = posterior probability that the email is spam
    spam_posterior = None

    '''END'''
    wgivenspamspamtotal=1
    wgivennotspamnotspamtotal=1
    content = text.split()
    
    for word in content:
        if word in spam_like_dict and word in ham_like_dict:
            wgivenspamspam = spam_like_dict[word]/ max(spam_like_dict.values()) * spam_prior
            wgivennotspamnotspam = ham_like_dict[word]/ max(ham_like_dict.values()) * ham_prior
            wgivenspamspamtotal *= Decimal(wgivenspamspam)
            wgivennotspamnotspamtotal *= Decimal(wgivennotspamnotspam)
    
    spam_posteriror = Decimal(wgivenspamspamtotal/ ((wgivenspamspamtotal + wgivennotspamnotspamtotal)))
    ham_posteriror = Decimal(wgivennotspamnotspamtotal/ ((wgivenspamspamtotal + wgivennotspamnotspamtotal)))
    
    if spam_posterior> ham_posterior:
        ham_spam_decision =1
    else:
        ham_spam_decision =0
    
    return ham_spam_decision



def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
	'''
	Calls "predict"
	'''
    hh = 0 #true negatives, truth = ham, predicted = ham
    hs = 0 #false positives, truth = ham, pred = spam
    sh = 0 #false negatives, truth = spam, pred = ham
    ss = 0 #true positives, truth = spam, pred = spam
    num_rows = df.shape[0]
    for i in range(num_rows):
        roi = df.iloc[i,:]
        roi_text = roi.text
        roi_label = roi.label_num
        guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
        if roi_label == 0 and guess == 0:
            hh += 1
        elif roi_label == 0 and guess == 1:
            hs += 1
        elif roi_label == 1 and guess == 0:
            sh += 1
        elif roi_label == 1 and guess == 1:
            ss += 1
    
    acc = (ss + hh)/(ss+hh+sh+hs)
    precision = (ss)/(ss + hs)
    recall = (ss)/(ss + sh)
    return acc, precision, recall
    
if __name__ == "__main__":
	'''YOUR CODE HERE'''
	#this cell is for your own testing of the functions above