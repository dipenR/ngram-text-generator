#!/usr/bin/python3
# DIPEN RUPANI 112321338
# CSE354, Spring 2021
##########################################################
## a3_Rupani_112321338.py

import sys
import re
import numpy as np
# import torch
# import torch.nn as nn

sys.stdout = open('a3_Rupani_112321338_OUTPUT.txt', 'w')

# loads data into a list from given filename
def loadData(filename):
    data = []
    with open(filename) as f:
        line = f.readline()
        while line:
            data.append(line.split('\t', 3)[2]) # split by tab but only 3 times for lemma, sense, context - then take just the context
            line = f.readline()

    return data

# receives a list of words and returns the same set of words back with 
    # <head> removed from target word and the index of head
def locateHead(words):
    headMatch=re.compile(r'<head>([^<]+)') # matches contents of head  
    for i in range(len(words)):
            m = headMatch.match(words[i])
            if m: #a match: we are at the target token
                words[i] = words[i].split('>')[1]
                return words

# creates a unigrams with given data
def createUnigrams(data, vocab = {}):
    for d in range(len(data)):
        words = [(wlp.split('/')[0]).lower() for wlp in data[d].split()]
        words = locateHead(words)
        words = ['<s>'] + words + ['</s>'] # prepend <s> and append </s>

        data[d] = " ".join(words)
        for word in words:
            vocab[word] = vocab.get(word, 0) + 1

    sorted_vocab = {}
    sorted_word_count = sorted(vocab, key = lambda k: (vocab[k], k))
    for w in sorted_word_count:
        sorted_vocab[w] = vocab[w]

    final_vocab = {k: sorted_vocab[k] for k in list(sorted_vocab)[-5000:]}
    temp_vocab = {k: sorted_vocab[k] for k in list(sorted_vocab)[:-5000]}
    final_vocab['<OOV>'] = sum(temp_vocab.values())

    # prep the contexts by removing all OOV words and replacing them with OOV
    for d in range(len(data)):
        words = data[d].split()
        for wi in range(len(words)):
            if words[wi] not in list(final_vocab.keys()):
                words[wi] = '<OOV>'
        data[d] = " ".join(words)

    return (data, final_vocab) # also unigram counts, just take keys for vocab

def createBigrams(data):
    bigrams = {}
    # print(data[:3])
    for d in data:
        words = d.split()
        for wi in range(len(words)-1):
            if words[wi] in bigrams.keys():
                bigrams[words[wi]][words[wi+1]] = bigrams[words[wi]].get(words[wi+1], 0) + 1
            else:
                bigrams[words[wi]] = {}
                bigrams[words[wi]][words[wi+1]] = bigrams[words[wi]].get(words[wi+1], 0) + 1

    return bigrams

def createTrigrams(data):
    trigrams = {}
    for d in data:
        words = d.split()
        for wi in range(len(words)-2):
            if (words[wi], words[wi+1]) in trigrams.keys():
                trigrams[(words[wi], words[wi+1])][words[wi+2]] = trigrams[(words[wi], words[wi+1])].get(words[wi+2], 0) + 1
            else:
                trigrams[(words[wi], words[wi+1])] = {}
                trigrams[(words[wi], words[wi+1])][words[wi+2]] = trigrams[(words[wi], words[wi+1])].get(words[wi+2], 0) + 1
    return trigrams

# probabilities are calculated as:
    # 1. bigramprob(w,w-1) = (bigramCount(w,w-1)+1)/(unigramCount(w-1)+V)
    # 2. trigrams(w,w-1,w-2) = (trigramCount(w,w-1,w-2)+1)/(BigramCount(w-1,w-2)+V)
       # trigramprob = (bigramprob(w,w-1)+trigrams(w,w-1,w-2))/2
def smoothedProbs(unigrams, bigrams, trigrams, wb1, wb2 = None):
    probs = {}
    # print(wb1)
    for word in bigrams[wb1]:
        bigram_probs = (bigrams[wb1][word] + 1)/(unigrams[wb1] + len(unigrams))
        if wb2:
            if word in trigrams[(wb2, wb1)]:
                trigram_probs = (trigrams[(wb2, wb1)][word] + 1)/(bigrams[wb2][wb1] + len(unigrams))
            else:
                trigram_probs = 1/(bigrams[wb2][wb1] + len(unigrams))
            probs[word] = (bigram_probs + trigram_probs)/2
        else:
            probs[word] = bigram_probs
    return probs

def generateNextWord(words, unigrams, bigrams, trigrams):
    if len(words) == 1:
        probs = smoothedProbs(unigrams, bigrams, trigrams, words[-1])    
    else:
        probs = smoothedProbs(unigrams, bigrams, trigrams, words[-1], words[-2])
    
    words = list(probs.keys()) # get all words
    sum_probs = sum(probs.values()) # to normalize
    pr = [p/sum_probs for p in probs.values()] # normalize
    nextWord = np.random.choice(np.array(words).flatten(), p=pr)
    return nextWord

def generateSentence(prompt, unigrams, bigrams, trigrams, max_length = 32):
    while (prompt[-1] != '</s>') and (len(prompt) != max_length):
        nextWord = generateNextWord(prompt, unigrams, bigrams, trigrams)
        prompt += [nextWord]
    return prompt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python3 a3_lastname_id.py onesec_train.tsv")
        sys.exit(1)
    train_file = sys.argv[1]

    train = loadData(train_file)

    train, unigrams = createUnigrams(train) # processes data to be more usable, creates unigram counts
    vocab = list(unigrams.keys())
    bigrams = createBigrams(train) # creates bigrams
    trigrams = createTrigrams(train)

    probs = smoothedProbs(unigrams, bigrams, trigrams, 'to')

    print_unis = ['language', 'the', 'process']
    print_bis = [['the','language'], ['<OOV>','language'], ['to','process']]
    print_tris = [[('specific', 'formal'), 'languages'], [('to', 'process'), '<OOV>'], [('specific', 'formal'), 'event']]

    print("\n\nCHECKPOINT 2.2 - COUNTS\n\n1-grams")
    for u in print_unis:
        try:
            print(f'({u})')
            print(unigrams[u])
        except KeyError:
            print(0)

    print("\n2-grams\n")
    for b in print_bis:
        # print(b)
        try:
            print(f"({b[0]}, {b[1]}):")
            print(bigrams[b[0]][b[1]])
        except KeyError:
            print(0)

    print("\n3-grams\n")
    for t in print_tris:
        # print(t)
        try:
            print(f"({t[0]}, {t[1]})")
            print(trigrams[t[0]][t[1]])
        except KeyError:
            print(0)
    
    print("\n\nCHECKPOINT 2.3 - SMOOTHED PROBABILITIES\n\n2-grams")
    print_bis = [['the', 'language'], ['<OOV>', 'language'], ['to', 'process']]
    print_tris = [['specific', 'formal', 'languages'], ['to', 'process', '<OOV>'], ['specific', 'formal', 'event']]

    for b in print_bis:
        try:
            print(f"({b[0]}, {b[1]}): ")
            print(smoothedProbs(unigrams, bigrams, trigrams, b[0])[b[1]])
        except KeyError:
            print("NOT VALID Wi\n")
    print("\n3-grams\n")
    for t in print_tris:
        try:
            print(f"({t[0]}, {t[1]}, {t[2]}): ")
            print(smoothedProbs(unigrams, bigrams, trigrams, t[1], t[0])[t[2]])
        except KeyError:
            print("NOT VALID Wi\n")
    
    print("\n\nFINAL CHECKPOINT: GENERATE LANGUAGE")
    prompts = ['<s>', '<s> language is', '<s> machines', '<s> they want to process']

    for p in prompts:
        print(f"\nprompt: {p}\n")
        print(" ".join(generateSentence(p.split(), unigrams, bigrams, trigrams)))
        print(" ".join(generateSentence(p.split(), unigrams, bigrams, trigrams)))
        print(" ".join(generateSentence(p.split(), unigrams, bigrams, trigrams)))
        print("\n")
    # print(prompts[0])
    # print(generateSentence(prompts[0].split(), unigrams, bigrams, trigrams))
    # print(generateSentence(prompts[0].split(), unigrams, bigrams, trigrams))
    # print(generateSentence(prompts[0].split(), unigrams, bigrams, trigrams))





