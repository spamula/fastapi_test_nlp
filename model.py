import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from pydantic import BaseModel

class SentimentQueryModel(BaseModel):
    text : str

class SentimentModel:
    def get_sentiment(self, text):
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('spacytextblob')
        doc = nlp(text)
        polarity = doc._.polarity      
        subjectivity = doc._.subjectivity
        return polarity, subjectivity