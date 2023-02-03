import pickle
from sentence_transformers import SentenceTransformer
import torch

# TODO: add support for prompt modifiers - generate embeddings 

def wordcloud(sentence, k=25):
    print("+++++ WordCloud started")

    nlp = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    #Load sentences & embeddings from disc
    with open('embeddings/wordnet_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        content_words = stored_data['words']
        word_embeddings = stored_data['embeddings']

    features = ['','objects','place','situation']
    emotions = ['Admiration','Adoration','Appreciation','Amusement','Excitement','Awe','Joy','Romance','Satisfaction','Interest','Surprise','Relief','Nostalgia','Sadness','Confusion','Calmness','Boredom','Desire','Craving','Fear','Anxiety','Anger']
    colors = ['Red','Orange','Yellow','Green','Blue','Indigo','Violet','Pink','Purple','Turquoise','Gold','Lime','Maroon','Navy','Coral','Teal','Brown','White','Black','Sky','Berry','Grey','Straw','Sapphire','Silver']

    top_k_content_words = []

    for feature in features:
      sentence_embedding = torch.tensor(nlp.encode(sentence+' '+feature), dtype=torch.float32)
      word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
      cosine_similarities = torch.nn.functional.cosine_similarity(sentence_embedding, word_embeddings, dim=1)
      top_k_indices = cosine_similarities.topk(k).indices.tolist()
      top_k_words = [content_words[i] for i in top_k_indices]
      print(feature+' related to '+sentence)
      print(top_k_words)
      print()
      top_k_content_words += top_k_words

    # emotions
    emotion_embeddings = torch.tensor(nlp.encode(emotions), dtype=torch.float32)
    cosine_similarities = torch.nn.functional.cosine_similarity(sentence_embedding, emotion_embeddings, dim=1)
    top_k_indices = cosine_similarities.topk(4).indices.tolist()
    top_k_words = [emotions[i] for i in top_k_indices]
    print('Emotions related to '+sentence)
    print(top_k_words)
    print()
    top_k_content_words += top_k_words

    # colors
    colors_embeddings = torch.tensor(nlp.encode(colors), dtype=torch.float32)
    cosine_similarities = torch.nn.functional.cosine_similarity(sentence_embedding,colors_embeddings, dim=1)
    top_k_indices = cosine_similarities.topk(4).indices.tolist()
    top_k_words = [colors[i] for i in top_k_indices]
    print('Colors related to '+sentence)
    print(top_k_words)
    top_k_content_words += top_k_words

    print()
    print('all words ', len(top_k_content_words))
    print(top_k_content_words)

    new_content_words = []
    for word in top_k_content_words:
        split_word = word.split("_")
        i = len(split_word)
        while i > 0:
          i -= 1
          new_content_words.append(split_word[i])     

    new_content_words = list(set(new_content_words))

    new_content_words = [word for word in new_content_words if not any(token in word for token in set(sentence.split()))]
    new_content_words = [word for word in new_content_words if word not in features]

    print()

    print("Word soup: ", new_content_words)

    return new_content_words