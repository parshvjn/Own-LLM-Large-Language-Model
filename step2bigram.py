#step2: bigram model: how often does each character follow another? (just counting how often certain charactesr come after another to predict 
# what woudl come next only by looking at the current character based on counts- no neural network yet)

from step1data import train, val, stoi, itos, encode, decode
import numpy as np

#row is current char, column is next character
counts = np.zeros((65, 65), dtype = int)
for x in range(len(train)-1):
    counts[train[x], train[x+1]] += 1
# print(counts)

#smoothing: removing 0 values to prevent infinite loss in future
counts = counts + 1

rowSums = counts.sum(axis = 1, keepdims = True, dtype= int)
P = counts / rowSums # converted to probabilities
# np.set_printoptions(suppress=True, precision=4)
# print(P)

#test: generate text
rng = np.random.default_rng(42) # same seed everytime, (same resutls everytime)

final = ""
current = "\n"
for x in range(300):
    row = P[stoi[current]]
    next = itos[rng.choice(65, p = row)]
    final+=next
    current = next

print(final)

#remember at this point, the model only sees one character back

#loss
total = 0.0
for x in range(len(val) - 1):
    currentC = val[x]
    nextC = val[x + 1]
    prob = P[currentC, nextC]
    total += -np.log(prob)

loss = total / (len(val) - 1)
print("validation loss:", loss)
print("random guess base (what to beat):", np.log(len(stoi))) # if it learnt nothing