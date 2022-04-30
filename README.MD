# API deployment

- Repository: `challenge-machine-learning-api-deployment`
- Type of Challenge: `Learning`
- Duration: `6 days`
- Deadline: `05/05/2022 04:30 PM` **(code)**
- Presentation: `06/05/2022 1:30 PM`
- Team challenge : Solo project

## Mission objectives

- Be able to deploy a machine learning model.
- Be able to create a Flask API that can handle a machine learning model.
- Deploy an API to Heroku.

## The Mission

The real estate company "ImmoEliza" is really happy about your data collection using web scrapping. Whenever a new property comes on the market, the question of how it should be priced naturally arises.

Now, the company asks you to create a machine learning model to predict prices on Belgium's real estate sales. "ImmoEliza" has hired you to build a tool that enables the company to predict property prices using linear regression.  

Take the dataset that was previously **scraped** and preprocess the data to be used with machine learning. 

In addition, they would like you to create an API to let their web-developers create a website around it.

Ideally, your API would ask a user to provide with information about a property (features) and return the estimated price using your model.


### Built With

* [Python](https://www.python.org/)
* [pycaret](https://pycaret.org//)
* [Catboost](https://catboost.ai//)
* [Scikit-learn](https://scikit-learn.org/)
* [Docker](https://www.docker.com/)
* [Heroku](https://www.heroku.com/)


## Preparation

In any API use case the first thing to decide _(for each route)_, is the **input** and the **output** you want.
Your very first step will be to decide that.

### Input Data
```
subtype
age
bedroomCount
bathroomCount
netHabitableSurface
toiletCount
transaction_certificates_epcScore
building_condition
kitchen_type
flags_isNewlyBuilt
hasBasement
hasDressingRoom
hasDisabledAccess
hasLift
hasArmoredDoor
hasVisiophone
hasSecureAccessAlarm
fireplaceExists
hasTerrace
transaction_sale_isFurnished
specificities_hasOffice
hasparking
```


### Output

Predicted price: same page as form
 

```json
{
  "prediction": Optional[float],
  "error": Optional[str]
}
```


### Must-have features

#### Step 1: Project preparation

- Create a folder to handle your project.
- Create a file `app.py` that will contain the code for your API.
- Create a folder `preprocessing/` that will contain all the code to preprocess your data.
- Create a folder `model/` that will contain your model.
- Create a folder `predict/` that will contain all the code to predict the price.

#### Step 2: Preprocessing pipeline

This python module will contain all the code to preprocess your data. Make sure to think about what will be the format of your data to fit the model.
Also, be sure to know which information HAVE to be there and which one can be empty (NAN).

In the `preprocessing/` folder:

- Create the `cleaning_data.py` file that will contain all the code that will be used to preprocess the data you will receive to predict a new price. (fill the NaN values, handle text data, etc...).
- This file should contain a function called `preprocess()` that will take a new house's data as input and return those data preprocessed as output.
- If your data doesn't contain the required information, you should return an error to the user.

#### Step 3: Fit your data

Fit your data to your model.

In the `predict/` folder:

- Create the `prediction.py` file that will contain all the code used to predict a new house's price.
- Your file should contain a function `train()` that will create and store the model and a function `predict()` that will take your preprocessed data as an input and return a price as output using your stored model.

#### Step 4: Create your API

In your `app.py` file, create a Flask API that contains:

- A route at `/` that accept:
  - `GET` request and return `"alive"` if the server is alive.
- A route at `/predict` that accept:
  - `POST` request that receives the data of a house in JSON format.
  - `GET` request returning a string to explain what the `POST` expect (data and format).

#### Step 5: Deploy your app on Heroku
Deploy your web app on Heroku using the requirements.txt file. You can use the following tutorial as a reference: [Deploying a Flask Application to Heroku](https://stackabuse.com/deploying-a-flask-application-to-heroku/) 

## Nice-to- have's:

- Try other regression models for your prediction. 
- Deploy your application using a docker image to Heroku. To deploy your API, using Docker:

  - Create a Dockerfile that creates an image with:
    - Ubuntu
    - Python 3.8
    - Flask
    - All the other dependencies you will need.
    - All the files of your project in an `/app` folder that you will previously create.
  - Create a `requirements.txt` file. 
  - Run your `app.py` file with Python.
  - Heroku will allow you to push your docker container on their server and start it.


## Deliverables

1. Pimp up the README file:
   - Description
   - Installation
   - Usage
   - (Visuals)
   - (Contributors)
   - (Timeline)
   - (Personal situation)
2. Your API is able to predict the price and is deployed on Heroku.
3. Pitch your project in 3 minutes. (NO SLIDES! DEMO ONLY!)

## Evaluation criteria

| Criteria       | Indicator                                                | Yes/No |
| -------------- | -------------------------------------------------------- | ------ |
| 1. Is complete | Your API works.                                          | [ ]    |
|                | The API is clear and the presentation is understandable. | [ ]    |
|                | README is pimped.                                        | [ ]    |
|                | Your model is trained and can predict a result.          | [ ]    |
|                | Your API is deployed on Heroku.              | [ ]    |
| 2. Is good     | The repo doesn't contain unnecessary files.              | [ ]    |
|                | You used typing.                                         | [ ]    |
|                | The presentation is clean.                               | [ ]    |


