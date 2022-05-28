import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    load_data
    Load data from csv files and merge to a single dataframe.
    
    Input: messages and categorgies filepaths
    Output: dataframe merging categories and messages.
    
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    
    '''
    clean_data
    Function for cleaning the dataframe merging categories and messages. 
    
    Input: dataframe with categories and messages.
    Outuput: clean dataframe with categories and messages.
   
    '''
    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = list(row.str.replace('-1', '').str.replace('-0', '').drop_duplicates())
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype('int64')
    
    for column in categories:
        categories[column] = np.where(categories[column]>=1, 1, 0)
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df


def save_data(df, database_filename):
    
    '''
    save_data
    Function to save the dataframe to a SQL database.
    
    Input: clean dataframe with categories and messages.
    Output: SQL database.
    
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_df', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()