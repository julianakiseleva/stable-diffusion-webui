import pickle
from sentence_transformers import SentenceTransformer
import torch

def wordcloud(sentence,k=40):
    print("+++++ WordCloud started")

    nlp = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Load sentences & embeddings from disc
    with open('embeddings/wordnet_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        content_words = stored_data['words']
        word_embeddings = stored_data['embeddings']

    sentence = sentence + " Emotions and feelings. Actions and objects. Situations and places."
    sentence_embedding = torch.tensor(nlp.encode(sentence), dtype=torch.float32)
    word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
    cosine_similarities = torch.nn.functional.cosine_similarity(sentence_embedding, word_embeddings, dim=1)

    top_k_indices = cosine_similarities.topk(k).indices.tolist()
    top_k_content_words = [content_words[i] for i in top_k_indices]
    print("Word soup: ",top_k_content_words)
    
    return top_k_content_words