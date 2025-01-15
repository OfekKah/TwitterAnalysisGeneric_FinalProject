import json
import torch
import transformers
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
import pandas as pd

config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

import gc
gc.collect()
torch.cuda.empty_cache()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    #device = 0,
    device_map="auto",
)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Initialize the text generation pipeline
with torch.cuda.device(0):  # Use GPU 0, if available
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.float16},  # Use float16 for better GPU performance
        device_map="auto",  # Automatically maps model layers to available devices (GPU if available)
    )

    # Test the pipeline with a sample input
    output = pipeline("Explain quantum computing in simple terms.")
    print(output)

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available. Using CPU.")


import sys
import sqlite3

if len(sys.argv) < 2:
    print("Usage: script.py <db_path>")
    sys.exit(1)

db_path = sys.argv[1]

# Connect to the database
conn = sqlite3.connect(db_path)

authors_table = 'authors'
all_author_data = pd.read_sql(f"SELECT * FROM {authors_table}", conn)
    
def extract_country_from_location(df, pipeline):
    # Function to apply to each location
    def process_location(location):
        if pd.isna(location):
            return 'NaN'
        prompt = [
        {
            "role": "system",
            "content": (
                "You are a country classifier. Given a string containing a location in various patterns, return the country only. "
                "Here are the rules you should follow to determine the country:\n"
                "1. If the location contains a comma-separated list, check each part sequentially and return the most accurate country name.\n"
                "2. For locations without a comma, determine the country by identifying its geographic placement.\n"
                "3. If the location is in abbreviations (e.g., 'NJ', 'UK'), convert it to the full country name.\n"
                "4. If the location is not a country or does not exist (e.g., 'Earth', 'Utopia'), return 'NaN'.\n"
                "5. If the location contains multiple countries, return the most relevant or prominent country:\n"
                "   - Prioritize the last mentioned country in sequential order unless context suggests otherwise.\n"
                "   - If no clear prominence exists, default to the last mentioned country.\n"
                "6. If the location is not written in English, identify the country and return the name in English.\n"
                "7. Always format country names with proper capitalization (e.g., 'Rwanda', 'United Kingdom').\n"
                "8. Return only the country name as the response, without additional explanations."
            )
        },
        {"role": "user", "content": f"give me the country of this location: {location}, answer only with the country name"}
    ]
        # Send the prompt to the Llama API
        response = pipeline(prompt, max_new_tokens=256)
        return response[0]["generated_text"][-1]['content']
    
    # Apply the function to the 'location' column
    df['country'] = df['location'].apply(process_location)
    return df[['location','country']]

author_locations = extract_country_from_location(all_author_data, pipeline)

import numpy as np
def clean_country_names(df):
    # Replace 'England' with 'United Kingdom'
    df['country'] = df['country'].replace('England', 'United Kingdom')
    
    # Replace 'unknown' with NaN
    df['country'] = df['country'].replace('Unknown', np.nan)
    
    return df

# Example usage:
author_locations_cleaned = clean_country_names(author_locations)
author_locations_cleaned.to_csv('author_locations_cleaned.csv', index=False)

# Merge the country information
all_author_data = all_author_data.merge(author_locations_cleaned[['location', 'country']], on='location', how='left')
all_author_data.to_csv('all_author_data.csv', index=False)

# Add the new 'country' column to the database
with conn:
    # Check if 'country' column already exists
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({authors_table})")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'country' not in columns:
        # Add the column if it doesn't exist
        cursor.execute(f"ALTER TABLE {authors_table} ADD COLUMN country TEXT")
    
    # Update the database with the new country data
    for index, row in all_author_data.iterrows():
        cursor.execute(
            f"UPDATE {authors_table} SET country = ? WHERE location = ?",
            (row['country_y'], row['location'])
        )

# Commit and close the connection
conn.commit()
conn.close()