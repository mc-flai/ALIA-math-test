############################################################################# 
# This script is valid for evaluating the filtered questions of the Valencian
# Country Kangur test for different levels. It can be used for gpt o3-mini and
# gpt-4o-mini. An API key is needed.
# Please, check the destination folders where responses will be saved as a txt
# file for each question.
#
#############################################################################

## Imports
import os
import pandas as pd
import time
from openai import OpenAI

## Read data 
df = pd.read_excel(r"ALIA-math-test/dataset/Dataset_Kangur_2015_2024_No_Figure.xlsx")

## Choose your gpt version
version = "gpt-4o-mini" # (or "o3-mini")
## Destination folder (the LLM responses will be saved here as individual txt files)
folder_path = r"ALIA-math-test/raw_responses/gpt-4o-mini/" # (or r"ALIA-math-test/raw_responses/o3-mini")
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")

## Configure gemini
client = OpenAI(api_key='your/api/key/here')

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
      chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        model = version,
        #temperature = 0,  # not available for o3-mini
        #top_p = 0 # not available for o3-mini         
      )

      response = chat_completion.choices[0].message.content.strip()
      print(response)

      # save the response
      with open(os.path.join(folder_path, f"{idx}.txt"), "w",encoding="utf-8") as f:
          f.write(f"{response}")
    except:
        # if generating the response fails, wait 3 seconds to repeat
        time.sleep(3)
        continue
