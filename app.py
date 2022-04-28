from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
water = 0
grain = 0
fodder = 0
prediction = []

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        clf = joblib.load("clf.pkl")
        soldier = request.form.get("soldier")
        mule = request.form.get("mule")
        slave = request.form.get("slave")
        X = pd.DataFrame([[soldier, mule, slave]], columns=["soldier", "mule", "slave"])
        prediction = clf.predict(X)[0]
        water = prediction[0]
        grain = prediction[1]
        fodder = prediction[2]
    else:
        water = 0
        grain = 0
        fodder = 0
    return render_template("website.html", water=water, grain=grain, fodder=fodder)
if __name__ == '__main__':
    app.run(host='0.0.0.0')