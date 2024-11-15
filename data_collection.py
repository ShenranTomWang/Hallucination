from groq import Groq
import pandas as pd
import os
import time

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MODEL_NAME = "llama3-8b-8192"
DATA_PATH = "./experimental_data/truthful_QA.csv"
START_IDX = 0
END_IDX = -1

data = pd.read_csv(DATA_PATH)
client = Groq(api_key="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")

count = START_IDX
end = END_IDX if END_IDX != -1 else len(data)
chat_completions = []
while count < end:
    print(count)
    # print(data["Question"][count])
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": data["Question"][count]
            }
        ],
        model=MODEL_NAME
    )
    data.loc[count, "single_agent_response"] = chat_completion.choices[0].message.content
    # print(chat_completion.choices[0].message.content)
    time.sleep(2)
    count += 1
    
if not os.path.exists("./experimental_data"):
    os.mkdir("./experimental_data")

data.to_csv("./experimental_data/truthful_QA.csv")