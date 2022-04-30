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
    _features=['subtype','age','bedroomCount','bathroomCount','netHabitableSurface','toiletCount','transaction_certificates_epcScore','building_condition','kitchen_type',
            'flags_isNewlyBuilt','hasBasement','hasDressingRoom','hasDisabledAccess','hasLift','hasArmoredDoor','hasVisiophone','hasSecureAccessAlarm','fireplaceExists',
            'hasTerrace','transaction_sale_isFurnished','specificities_hasOffice','hasparking']
    _list=[]
    for key in _features:
        if key in _a:
            _list.append(_a[key])
        else:
            _list.append(0)

    pred_array = np.array(_list)
    
    return pred_array
    


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        clf = joblib.load("clf.pkl")
        _alldata=request.form.to_dict()
        _q=clean_data(_alldata)
        prediction=clf.predict(_q)
        prediction="{0:,.2f}".format(prediction)
        prediction = "It is around " + str(prediction) + " Euros" 
       
    else:
        prediction = ""
        
    return render_template("website.html", prediction=prediction )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)