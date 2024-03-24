from flask import Flask, jsonify

app = Flask(__name__)

# Define a route for an API endpoint
@app.route('/api/hello')
def hello():
    # Return JSON response
    return jsonify({'message': 'Hello, World!'})

@app.route('/api/hello_1')
def other_helper():
    return jsonify({'message': 'test'})

if __name__ == '__main__':
    app.run(debug=True)
