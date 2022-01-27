import uvicorn
from fastapi import FastAPI
from model import SentimentModel, SentimentQueryModel

app = FastAPI()
model = SentimentModel()

@app.post('/predict')
def predict(data: SentimentQueryModel):
    data = data.dict()
    polarity, subjectivity = model.get_sentiment(data['text'])
    return {
        'polarity': polarity,
        'subjectivity': subjectivity
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)