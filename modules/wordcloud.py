import pickle
from sentence_transformers import SentenceTransformer
import torch

# TODO: add support for prompt modifiers - generate embeddings 

def wordcloud(sentence, k = 60):
    print("+++++ WordCloud started")

    nlp = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Load sentences & embeddings from disc
    with open('embeddings/wordnet_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        content_words = stored_data['words']
        word_embeddings = stored_data['embeddings']

    sentence = "Creative art design for '" + sentence + "' in relation to emotions and feelings, actions and intentions, objects and places."
    sentence_embedding = torch.tensor(nlp.encode(sentence), dtype=torch.float32)
    word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
    cosine_similarities = torch.nn.functional.cosine_similarity(sentence_embedding, word_embeddings, dim=1)

    top_k_indices = cosine_similarities.topk(k).indices.tolist()
    top_k_content_words = [content_words[i] for i in top_k_indices]

    new_content_words = []
    for word in top_k_content_words:
        split_word = word.split("_")
        i = len(split_word)
        while i > 0:
          i -= 1
          new_content_words.append(split_word[i])     

    new_content_words = list(set(new_content_words))

    # remove prompt words from word cloud
    new_content_words = [word for word in new_content_words if not any(token in word for token in set(sentence.split()))]

    print("Word soup: ", new_content_words)
    
    return new_content_words