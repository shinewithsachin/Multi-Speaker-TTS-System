import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class NLPProcessor:
    def process_text(self, text):
        tokens = word_tokenize(text)
        filtered = [t for t in tokens if t.lower() not in stopwords.words('english')]
        return " ".join(filtered)
