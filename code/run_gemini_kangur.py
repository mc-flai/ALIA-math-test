############################################################################# 
# This script is valid for evaluating the filtered questions of the Valencian
# Country Kangur test for different levels. It can be used for gemini-1.5-flash
# and gemini-2.0-flash-exp. An API key is needed (https://aistudio.google.com/apikey).
# Please, check the destination folders where responses will be saved as a txt
# file for each question.
#
#############################################################################

## Imports
import os
import pandas as pd
import re
import time
import google.generativeai as genai

## Read data 
df = pd.read_excel(r"ALIA-math-test/dataset/Dataset_Kangur_2015_2024_No_Figure.xlsx")

## Choose your gemini version
version = "gemini-1.5-flash" # (or 'gemini-2.0-flash-exp')
## Destination folder (the LLM responses will be saved here as individual txt files)
folder_path = r"ALIA-math-test/raw_responses/gemini-1v5-flash/" # (or r"ALIA-math-test/raw_responses/gemini-2v0-flash-exp")
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")

## Configure gemini
GOOGLE_API_KEY = "your/api/key/here"
genai.configure(api_key = GOOGLE_API_KEY)
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
temperature = 0
top_p = 0
top_k = 1
generation_config = genai.GenerationConfig(temperature=temperature, top_p=top_p,top_k=top_k)
model = genai.GenerativeModel(version,
                              generation_config=generation_config,safety_settings=safety_settings)

## Loop: 1 API call per question
for index, cell in df["question"].items():
    # track the original indexes
    idx = int(df.loc[index,"original index"])
    print(idx)
    # extract the questions and the possible responses
    question = cell
    answer = df.loc[index,"outputs"]
    # skip questions if they have a figure
    if df.loc[index,"figure"] == "YES":
       continue
    # One path per index in the df, if it exists, skip it to save resources
    file_path = os.path.join(folder_path,f"{idx}.txt")
    if os.path.exists(file_path):
      print(f"Skipping {file_path}")
      continue
    # Structured prompt
    prompt = f""" 
    Pregunta: {cell}
    Opcions de resposta: {answer}

    Instruccions: Raona com has arribat a la resposta correcta i proporciona la teua resposta en aquest format:
      Raonament: Explica el procés de raonament per arribar a la resposta.
      Resposta: Escriu una de les opcions: A), B), C), D) o E).
    """
    # check prompt
    print(prompt)

    try:
      # generate the response
      response = model.generate_content(prompt)
      print(f"Response: {response.text}")
      # save the response
      with open(os.path.join(folder_path, f"{idx}.txt"), "w",encoding="utf-8") as f:
          f.write(f"{response.text}")
    except:
        # if generating the response fails, wait 3 seconds to repeat
        time.sleep(3)
        continue
