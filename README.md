# Udacity P2: Disaster Response Pipeline

## Libraries
- Pandas.
- Numpy.
- Sklearn.
- Re.
- Nltk.
- Sqlalchemy.
- Pickle.

## Project motivation
The purpose of this project consist in analyze disaster data from Appen to build a model for an API that classifies disaster messages. In this way, this project uses a data set containing real messages that were sent during disaster events and with a machine learning pipeline categorize these events so an emergency worker 
can send the messages to an appropriate disaster relief agency.

## File descriptions
- Data: folder with the datasets messages.csv and categories.csv.
- [ETL Pipeline Preparation.ipynb](https://github.com/gutorres93/udacity_p2/blob/main/ETL%20Pipeline%20Preparation.ipynb): a python script with the data cleaning pipeline.
- [ML Pipeline Preparation.ipynb](https://github.com/gutorres93/udacity_p2/blob/main/ML%20Pipeline%20Preparation.ipynb): a  python script with the machine learning pipeline.
- Web app: folder with the files to run the web app.

  To create a processed sqlite db:
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  
  To train and save a pkl model:
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl  
  
  To deploy the application locally:
  python run.py

## Licensing, Authors, Acknowledgements

The author of this project is Gustavo Torres, economist from Universidad de los Andes, currently working as a Senior Data Scientist at Banco Davivienda. Must give credit to Appen for the data.
