from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)   

try:
    model = joblib.load('twitter_sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print(" Model loaded successfully!")
except Exception as e:
    print(" Error loading model:", e)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweet = data.get('tweet', '').strip()
        
        if not tweet:
            return jsonify({'error': 'No tweet provided'}), 400
        
        cleaned_tweet = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned_tweet])
        
        # Get model prediction
        ml_prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        ml_confidence = float(max(probability) * 100)

        lower_tweet = tweet.lower()

        # ================== IMPROVED NEGATIVE DETECTION ==================
        strong_negative = ['hate', 'terrible', 'awful', 'worst', 'horrible', 'disgusting', 
                          'pathetic', 'useless', 'stupid', 'idiot', 'never', 'waste', 
                          'bad', 'sucks', 'ruined', 'angry', 'frustrated']

        strong_positive = ['love', 'amazing', 'awesome', 'fantastic', 'excellent', 'best', 
                          'wonderful', 'perfect', 'happy', 'great']

        neg_count = sum(1 for word in strong_negative if word in lower_tweet)
        pos_count = sum(1 for word in strong_positive if word in lower_tweet)

        if neg_count >= 1 and neg_count > pos_count:
            sentiment = "Negative"
            confidence = max(78, ml_confidence)
        elif pos_count >= 1 and pos_count > neg_count:
            sentiment = "Positive"
            confidence = max(78, ml_confidence)
        else:
            # If no strong keywords, trust the model more
            sentiment = "Positive" if ml_prediction == 1 else "Negative"
            confidence = ml_confidence

        # Final adjustment for sarcasm / mild cases
        if "not good" in lower_tweet or "not great" in lower_tweet or "quite bad" in lower_tweet:
            sentiment = "Negative"
            confidence = 82

        stars = round((confidence / 100) * 5)

        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 1),
            'stars': stars,
            'tweet': tweet
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)