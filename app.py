from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os
app = Flask(__name__)
prediction = ""

def clean_data(_a:dict):
    '''
        Function to convert recevided data from web form to numpp array
        All values are coming in a list so it is not need to check each one if they are valid. even if there is a data leak
        function wont send them to prediction.
    '''
    print(_a)
    _features=['subtype','age','bedroomCount','bathroomCount','netHabitableSurface','toiletCount','transaction_certificates_epcScore','building_condition','kitchen_type',
            'flags_isNewlyBuilt','hasBasement','hasDressingRoom','hasDisabledAccess','hasLift','hasArmoredDoor','hasVisiophone','hasSecureAccessAlarm','fireplaceExists',
            'hasTerrace','transaction_sale_isFurnished','specificities_hasOffice','hasparking']
    _list=[]
    for key in _features:
        if key in _a:
            _list.append(_a[key])
        else:
            _list.append(0)
    return np.array(_list)
    
@app.route('/', methods=['GET', 'POST'])
def main():
    '''
        Function to get data from form and return the prediction in the same template. 
    '''
    if request.method == "POST":
        clf = joblib.load("clf.pkl")
        prediction="{0:,.2f}".format(clf.predict(clean_data(request.form.to_dict())))
        prediction = "It is around " + str(prediction) + " Euros" 
    else:
        prediction = ""
    return render_template("website.html", prediction=prediction )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
