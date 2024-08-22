import os
import math
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

TARGET_LANGUAGE = "eng_Latn"
# define maximum length for each chunk
MAX_LENGTH = 512
# MODEL_NAME = 'google/madlad400-10b-mt'
# MODEL_NAME = 'google/madlad400-3b-mt'
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# enable fallback to CPU for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = 0 if torch.cuda.is_available() else -1

# initialize translation model
translator = pipeline("translation", model=MODEL_NAME, device=device, max_length=MAX_LENGTH)


def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def create_chunks(sentences, max_length):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    # add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def translate_text(text, translator, src_lang):
    if pd.isna(text):
        return text
    try:
        cleaned_text = clean_text(text)

        # do not translate from/to same language
        if src_lang == TARGET_LANGUAGE:
            return cleaned_text

        sentences = sent_tokenize(cleaned_text)

        # create chunks of sentences
        # soft_max_length = math.floor(0.9 * MAX_LENGTH)
        # chunks = create_chunks(sentences, soft_max_length)

        # translate each chunk
        translations = [translator(sentence, src_lang=src_lang, tgt_lang=TARGET_LANGUAGE)[0]['translation_text'] for sentence in tqdm(sentences, desc="translating sentences...")]

        # join translated chunk
        translated_text = ' '.join(translations)

        return translated_text
    except Exception as e:
        print(f"error translating text: {text[:30]}...: {e}")
        return text  # Return original text if translation fails


def translate_csv_file(input_path, output_path, translator, src_lang):
    try:
        print(f'\n\ntranslating from {src_lang} to {TARGET_LANGUAGE} in {input_path}')

        # read the tab-delimited csv file
        df = pd.read_csv(input_path, sep='\t')

        # translate `ils._title` and `ils.body` with progress bar
        tqdm.pandas(desc=f'translating titles from {src_lang} to {TARGET_LANGUAGE}')
        df['ils._title_translated'] = df['ils._title'].progress_apply(lambda x: translate_text(x, translator, src_lang))

        tqdm.pandas(desc=f'translating bodies from {src_lang} to {TARGET_LANGUAGE}')
        df['ils.body_translated'] = df['ils.body'].progress_apply(lambda x: translate_text(x, translator, src_lang))

        # save the translated csv file
        df.to_csv(output_path, sep='\t', index=False)
        return output_path
    except pd.errors.ParserError as e:
        print(f"error parsing {input_path}: {e}")
        return None


def translate_all_csv_files(input_path, output_path, translator):
    csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv") and '_translated' not in f]
    for filename in tqdm(csv_files, desc="translating csv files"):
        src_lang = filename
        filepath = os.path.join(input_path, filename)
        output_filepath = os.path.join(output_path, filename).replace(".csv", "_translated.csv")
        new_filepath = translate_csv_file(filepath, output_filepath, translator, src_lang)
        if new_filepath:
            print(f"translated file saved to: {new_filepath}")


# get the folder path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'data')
output_path = os.path.join(script_dir, 'results')

# translate all csv files in the folder
translate_all_csv_files(input_path, output_path, translator)
