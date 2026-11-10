#step1: tokenizer -- assigning each unique character to a number to encode into arrays with

import numpy as np

with open("data/input.txt", "r", encoding="utf-8") as text:
    content = text.read()
    # print("length: ", len(content))
    # text.seek(0)
    # print(text.read(300))

# print(content)
characters = sorted(set(content))
# print(characters)
# print(len(characters), ": characters")

# {str : int}
# {int : str}
stoi, itos = {}, {}
for i, key in enumerate(characters):
    stoi[key] = i
    itos[i] = key

def encode(text):
    encoded = []
    for char in text:
        encoded.append(stoi[char])
    return encoded

def decode(ids):
    string = ""
    for i in ids:
        string += itos[i]
    return string

# encoded = encode("Hi my name is Parshv J")
# print(encoded)
# print(decode(encoded))

#whole text encoded
enW = np.array(encode(content), dtype = int)
# print(enW.shape)
# print(enW[0:21])
n = int(len(enW)*0.9)
train = enW[:n]
val = enW[n:]
# print(len(train), len(test))