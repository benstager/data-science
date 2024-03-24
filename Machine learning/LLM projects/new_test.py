from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a route to handle incoming text data and return sentiment predictions
@app.route('/api/predict_sentiment', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        # Get the text data from the request
        data = request.json
        
        # Perform sentiment analysis using vaderSentiment
        scores = analyzer.polarity_scores(data['text'])
        
        # Determine the overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Return the predicted sentiment as a JSON response
        return jsonify({'sentiment': sentiment})
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)