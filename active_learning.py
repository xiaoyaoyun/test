#!/usr/bin/env python


import torch 
import math
from random import shuffle

class UncertaintySampling():
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def least_confidence(self, prob_dist, sorted=False):
        if sorted:
            simple_least_conf = prob_dist.data[0] # most confident prediction
        else:
            simple_least_conf = torch.max(prob_dist) # most confident prediction
        num_labels = prob_dist.numel() # number of labels
        normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1))
        return normalized_least_conf.item()
    
    def margin_confidence(self, prob_dist, sorted=False):
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True) # sort probs so largest is first
        difference = (prob_dist.data[0] - prob_dist.data[1]) # difference between top two props
        margin_conf = 1 - difference 
        return margin_conf.item()
        
    def ratio_confidence(self, prob_dist, sorted=False):
        if not sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True) # sort probs so largest is first
        ratio_conf = prob_dist.data[1] / prob_dist.data[0] # ratio between top two props
        return ratio_conf.item()
    
    def entropy_based(self, prob_dist):
        log_probs = prob_dist * torch.log2(prob_dist) # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs)
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
        return normalized_entropy.item()
   
    def softmax(self, scores, base=math.e):
        exps = (base**scores.to(dtype=torch.float)) # exponential for each value in array
        sum_exps = torch.sum(exps) # sum of all exponentials
        prob_dist = exps / sum_exps # normalize exponentials 
        return prob_dist
        
   
        
        
    def get_samples(self, model, unlabeled_data, method, feature_method, number=5, limit=10000):
    
        samples = []
    
        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose: # we're drawing from *a lot* of data this will take a while                                               
            print("Get predictions for a large amount of unlabeled data: this might take a while")
        else:
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
        with torch.no_grad():
            v=0
            for item in unlabeled_data:
                text = item[1]
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  
                prob_dist = torch.exp(log_probs) # the probability distribution of our prediction
                score = method(prob_dist.data[0]) # get the specific type of uncertainty sampling
                item[3] = method.__name__ # the type of uncertainty sampling used 
                item[4] = score
                samples.append(item)
                
        samples.sort(reverse=True, key=lambda x: x[4])       
        return samples[:number:]        
        
    

        