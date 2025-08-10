from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]
    label = "Spam" if prediction == 1 else "Not Spam"
    return render_template("index.html", prediction=label, input_message=message)

if __name__ == "__main__":
    app.run(debug=True)
