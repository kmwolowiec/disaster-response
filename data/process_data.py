import sys
import pandas as pd
import sqlite3


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """Load and preprocess messages and labels.

    Load messages and categories from .csv files.
    Split categories into separate columns.
    Merge messages and categories.

    :param messages_filepath path to messages .csv
    :param categories_filepath path to categories .csv
    :return pd.DataFrame
    """

    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(categories, how='left', on='id')
    categories = df.categories.str.split(';', expand=True)
    colnames = [colname.split('-')[0] for colname in df.categories.sample().str.split(';').tolist()[0]]
    categories.columns = colnames

    for column in categories:
        categories[column] = categories[column].str.slice(start=-1)
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def clean_data(df):

    # 'related' label has 3 unique values: 0, 1, 2
    # The '2' comes from uncorrect mapping and should be replaced by '0'
    # According to: https://knowledge.udacity.com/questions/64417
    df.loc[df['related'] == 2, 'related'] = 0

    # 'child_alone' label has only one unique value: 0 so it's useless
    df = df.drop('child_alone', axis=1)

    return df


def save_data(df, database_filename):
    conn = sqlite3.connect(f'{database_filename}')
    df.to_sql('dataset', conn, if_exists='replace', index=False)


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