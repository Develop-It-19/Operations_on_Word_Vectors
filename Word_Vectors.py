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
  norm_u = np.sqrt(np.sum(np.power(u, 2)))
  norm_v = np.sqrt(np.sum(np.power(v, 2)))
  cosine_similarity = np.divide(dot, norm_u * norm_v)
  
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
    
    cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
    
    if cosine_sim > max_cosine_sim:
      max_cosine_sim = cosine_sim
      best_word = w
  
  return best_word

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
  print('{} -> {} :: {} -> {}' .format(*triad, complete_analogy(*triad, word_to_vec_map)))
  
#Debiasing Word Vectors
g1 = word_to_vec_map['woman'] - word_to_vec_map['man']
g2 = word_to_vec_map['mother'] - word_to_vec_map['father']
g3 = word_to_vec_map['girl'] - word_to_vec_map['boy']
g4 = word_to_vec_map['king'] - word_to_vec_map['queen']
g = np.divide(np.add(g1, g2, g3, g4), 4)
print(g)

print("List of names and their similarities with constructed vector:")

name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

print("Other words and their similarities:")
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print(w, cosine_similarity(word_to_vec_map[w], g))

#Neutralize Bias for Non-gender specific words.
def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map["word"]
    e_biascomponent = np.divide(np.dot(e, g), np.linalg.norm(g) ** 2) * g    #Projection of e onto the direction g.
                                                                             #Projection of a on b = ((a.b)/(|b|^2))*b
    e_debiased = e - e_biascomponent
    
    return e_debiased

e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))
    #cos(theta) = 0     so theta = 90 degrees

#Equalization Algorithm for Gender-specific words.
def equalize(pair, bias_axis, word_to_vec_map):
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map["w1"], word_to_vec_map["w2"]
    
    mu = np.divide(np.add(e_w1, ew2), 2)
    
    mu_B = np.divide(np.dot(mu, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
    mu_orth = mu - mu_B
    
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.linalg.norm(bias_axis) ** 2) * bias_axis
    
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * np.divide(e_w1B - mu_B, np.abs(e_w1 - mu_orth - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * np.divide(e_w2B - mu_B, np.abs(e_w2 - mu_orth - mu_B)
    
    e1 = mu_orth + corrected_e_w1B
    e2 = mu_orth + corrected_e_w2B
    
    return e1, e2
                                                                                
print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))

