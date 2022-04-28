
import pandas as pd
import joblib

#app = Flask(__name__)
water = 0
grain = 0
fodder = 0
prediction = []



clf = joblib.load("clf.pkl")

print(clf)

# soldier = request.form.get("soldier")
# mule = request.form.get("mule")
# slave = request.form.get("slave")
# X = pd.DataFrame([[solcd..dier, mule, slave]], columns=["soldier", "mule", "slave"])
# prediction = clf.predict(X)[0]
# water = prediction[0]
# grain = prediction[1]
# fodder = prediction[2]
# else:
# water = 0
# grain = 0
#         fodder = 0
    


