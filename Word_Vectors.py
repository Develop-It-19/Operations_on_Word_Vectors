# Load Pre-trained Word Vectors.
    #Because Word Embeddings are Very Computationally Expensive to Train.
# Measure similarity using Cosign Similarity
# Solve Word Analogy Problems using Word Embeddings.
# Reduce Gender Bias by Modifying Word Embeddings.

#Import Dependencies
import numpy as np
from w2v_utils import *

#Load the 50-dimensional GloVe to represent words.
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

#Cosine Similarity between two Word Vectors.
def cosine_similarity(u, v):
  distance = 0.0
  dot = np.dot(u, v)
  norm_u = np.sqrt(np.sum(u * u))
  norm_v = np.sqrt(np.sum(v * v))
  cosine_similarity = dot / (norm_u * norm_v)
  
  return cosine_similarity

father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

# Word Analogy Task
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
  word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
  e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
  
  words = word_to_vec_map.keys()
  max_cosine_sim = -100
  best_word = None
  
  for w in words:
    if w in [word_a, word_b, word_c]:
      continue
    
    cosine-sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
    
    if cosine_sim > max_cosine_sim:
      max_cosine_sim = cosine_sim
      best_word = w
  
  return best_word

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
  print('{} -> {} :: {} -> {}' .format(*triad, complete_analogy(*triad, word_to_vec_map)))
  
#Debiasing Word Vectors

    











