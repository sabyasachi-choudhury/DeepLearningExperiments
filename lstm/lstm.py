import json
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

with open("ild.text", "r", encoding='utf-8') as file:
    data = file.read()

data = re.sub(r"\n", ' ', data)
data = re.findall(r"(Scepticism is.*)CONCLUDING NOTE", data)[0]
data = re.sub(r"\[\d+\]", "", data)
data = re.sub(r"\[Illustration: \]", "", data).lower()
data = re.sub(r"[,;:?!“”'’\-_+=\[\]<>/&*^%$()#@\"]+", "", data)
data = re.sub(r"\s+", " ", data)
data = data.split(".")
n_data = []
for i, elem in enumerate(data):
    data[i] = re.sub(r"\(.*\)", " ", data[i])
    data[i] = re.sub(r"^\s", "", data[i])
    data[i] = re.sub(r"\s$", "", data[i])
    if len(data[i].split(" ")) >= 5:
        n_data.append(data[i])
data = n_data

tokenizer = Tokenizer(num_words=7500)
tokenizer.fit_on_texts(data)

print(tokenizer.get_config().keys())
print(tokenizer.get_config()['word_index'])

all_vecs = {}
with open("word_vectors.vec", "r", encoding='utf-8') as file:
    words, l = file.readline().split(" ")
    print(words, l)
    for line in file:
        line = line.split(" ")
        all_vecs[line[0]] = np.array(line[1:], dtype=np.float16)
    print(all_vecs)

c = tokenizer.get_config()['word_index']
fin_vecs = {word: all_vecs[word] for word in c if all_vecs.get(word) is not None}
with open("doc_vecs.json", "w") as file:
    json.dump(fin_vecs, file)