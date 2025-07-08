from flask import Flask, render_template, request
from model.recommender import recommend

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    product_name = request.form["product"]
    recommendations = recommend(product_name)
    return render_template("result.html", product=product_name, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
