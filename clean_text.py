import os
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def clean_text(text):
    if not isinstance(text, str):
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def clean_csv_file(input_path, output_path, src_lang):
    try:
        print(f'\n\ncleaning {src_lang} in {input_path}')

        # read the tab-delimited csv file
        df = pd.read_csv(input_path, sep='\t')

        # check if required columns exist
        if 'ils._title' not in df.columns or 'ils.body' not in df.columns:
            print(f"Columns 'ils._title' or 'ils.body' missing in {input_path}")
            return None

        # translate `ils._title` and `ils.body` with progress bar
        tqdm.pandas(desc='cleaning titles')
        df['ils._title_translated'] = df['ils._title'].progress_apply(lambda x: clean_text(x))

        tqdm.pandas(desc='cleaning bodies')
        df['ils.body_translated'] = df['ils.body'].progress_apply(lambda x: clean_text(x))

        # save the translated csv file
        df.to_csv(output_path, sep='\t', index=False)
        return output_path
    except pd.errors.ParserError as e:
        print(f"Error parsing {input_path}: {e}")
        return None
    except KeyError as e:
        print(f"KeyError: {e}")
        return None


def clean_all_csv_files(input_path, output_path):
    csv_files = [f for f in os.listdir(input_path) if f.endswith("eng_Latn.csv") and '_cleaned' not in f]
    for filename in tqdm(csv_files, desc="cleaning csv files"):
        src_lang = filename
        filepath = os.path.join(input_path, filename)
        output_filepath = os.path.join(output_path, filename).rpartition(".csv")[0] + "_clean.csv"
        new_filepath = clean_csv_file(filepath, output_filepath, src_lang)
        if new_filepath:
            print(f"cleaned file saved to: {new_filepath}")


# get the folder path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'data')
output_path = os.path.join(script_dir, 'results')

# translate all csv files in the folder
clean_all_csv_files(input_path, output_path)
