from sentence_transformers import SentenceTransformer

class Vectorizer():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_vector(self, sentence):
        return self.model.encode(sentence, convert_to_numpy=True)
