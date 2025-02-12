############################################################################# 
# This script is valid for evaluating the filtered questions of the Valencian
# Country Kangur test for different levels. It can be used for any DeepSeek-R1
# version. Installing Ollama and downloading the models are required.
# https://ollama.com/library/deepseek-r1 (all available Distill-Qwen versions
# and original model). Note that deepseek-r1:671b is 404 Gb and even with 
# quantization, you need a powerful machine
# Please, check the destination folders where responses will be saved as a txt
# file for each question.
#
#############################################################################

## Imports
from ollama import chat
from ollama import ChatResponse
import pandas as pd
import os

## Read data 
df = pd.read_excel(r"ALIA-math-test/dataset/Dataset_Kangur_2015_2024_No_Figure.xlsx")

## Choose your deepseek version
version = "deepseek-r1:1.5b" # (or "deepseek-r1:7b", "deepseek-r1:8b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "deepseek-r1:671b")
## Destination folder (the LLM responses will be saved here as individual txt files)
folder_path = r"ALIA-math-test/raw_responses/deepseek-r1-7B/" # (or deepseek-r1-8B, deepseek-r1-14B,...,deepseek-r1-671B)
if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created {folder_path}")

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
      response: ChatResponse = chat(model=version, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
      print(response['message']['content'])
      # save the response
      with open(os.path.join(folder_path, f"{idx}.txt"), "w",encoding="utf-8") as f:
          f.write(f"{response['message']['content']}")
    except:
        # if generating the response fails, wait 3 seconds to repeat
        time.sleep(3)
        continue
