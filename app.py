# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load

# Load the model
spam_clf = load(open('./models/spam_detector_model.pkl','rb'))

# Load vectorizer
vectorizer = load(open('./vectors/vectorizer.pickle', 'rb'))


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}


# Define the route to the sentiment predictor
@app.post("/predict_sentiment")
def predict_sentiment(text_message):

    polarity = ""

    if(not(text_message)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid text message")

    prediction = spam_clf.predict(vectorizer.transform([text_message]))

    if(prediction[0] == 0):
        polarity = "Ham"

    elif(prediction[0] == 1):
        polarity = "Spam"
        
    return {
            "text_message": text_message, 
            "sentiment_polarity": polarity
           }