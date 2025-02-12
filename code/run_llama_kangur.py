############################################################################# 
# This script is valid for evaluating the filtered questions of the Valencian
# Country Kangur test for different levels. It can be used for Llama-3.1-8B-Instruct,
# Llama-3.2-3B-Instruct, and Llama-3.3-70B-Instruct. Requesting access is needed.
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
# Please, check the destination folders where responses will be saved as a txt
# file for each question. 
#
#############################################################################

## Imports
from transformers import pipeline
import torch
import pandas as pd
import os
import time

## Read data 
df = pd.read_excel(r"ALIA-math-test/dataset/Dataset_Kangur_2015_2024_No_Figure.xlsx")

## Model path
model_id = "/path/to/your/Llama-3.1-8B-Instruct" # or Llama-3.2-3B-Instruct or Llama-3.3-70B-Instruct
## Destination folder (the LLM responses will be saved here as individual txt files)
folder_path = r"ALIA-math-test/raw_responses/llama-8B-it/" # (or llama-3B-it or llama-3B-it)
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")

## Pipeline
pipe = pipeline(
    "text-generation",
    model = model_id,
    model_kwargs = {"torch_dtype": torch.bfloat16},
    device="cuda",
    #device_map = "auto", # activate this option if you do not have enough GPU
)

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
      messages = [
        {"role": "user", "content": f"{prompt}"},
      ]

      outputs = pipe(messages,
               max_new_tokens = 2048,
               do_sample = False,)

      response = assistant_response = outputs[0]["generated_text"][-1]["content"]
      # save the response
      with open(os.path.join(folder_path, f"{idx}.txt"), "w",encoding="utf-8") as f:
          f.write(f"{response}")
    except:
        # if generating the response fails, wait 3 seconds to repeat
        time.sleep(3)
        continue
