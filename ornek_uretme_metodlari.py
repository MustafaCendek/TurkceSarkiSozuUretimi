import numpy as np
import tensorflow as tf

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def greedy_search(conditional_probability):
    return (np.argmax(conditional_probability))

def temperature_sampling(conditional_probability, temperature=1.0):
    conditional_probability = np.asarray(
        conditional_probability).astype("float64")
    conditional_probability = np.log(conditional_probability) / temperature
    reweighted_conditional_probability = softmax(conditional_probability)
    probas = np.random.multinomial(1, reweighted_conditional_probability, 1)
    return np.argmax(probas)

def top_k_sampling(conditional_probability, k):
    top_k_probabilities, top_k_indices = tf.math.top_k(
        conditional_probability, k=k, sorted=True)
    top_k_probabilities = np.asarray(top_k_probabilities).astype("float32")
    top_k_probabilities = np.squeeze(top_k_probabilities)
    top_k_indices = np.asarray(top_k_indices).astype("int32")
    top_k_redistributed_probability = softmax(top_k_probabilities)
    top_k_redistributed_probability = np.asarray(
        top_k_redistributed_probability).astype("float32")
    sampled_token = np.random.choice(np.squeeze(
        top_k_indices), p=top_k_redistributed_probability)
    return sampled_token