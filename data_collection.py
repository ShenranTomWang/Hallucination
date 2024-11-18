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
    question = data["Question"][count]
    answer1 = data["single_agent_response"][count]
    refiner_prompt1 = f"{question}. Here is a response to this question: {answer1}. Please provide feedback on this answer, being specific to the question and providing actionable feedback. Ask 2 questions regarding the aspects of the response that seem incorrect."
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": refiner_prompt1
            }
        ],
        model=MODEL_NAME
    )
    time.sleep(2)
    
    feedback1 = chat_completion.choices[0].message.content
    proposer_prompt2 = f"{question} Here is an initial response {answer1} and a feedback {feedback1}. Please refine this response, considering the feedback and questions."
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": proposer_prompt2
            }
        ],
        model=MODEL_NAME
    )
    time.sleep(2)
    
    answer2 = chat_completion.choices[0].message.content
    refiner_prompt2 = f"{question}. Here is the initial response: {answer1}, initial feedback: {feedback1}, and the refined answer: {answer2}. Please provide feedback on this answer, being specific to the question and providing actionable feedback. Ask 2 questions regarding the aspects of the response that seem incorrect."
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": refiner_prompt2
            }
        ],
        model=MODEL_NAME
    )
    time.sleep(2)
    
    feedback2 = chat_completion.choices[0].message.content
    proposer_prompt3 = f"{question} Here is an initial response {answer1} and an initial feedback {feedback1}. The response is then refined to {answer2}, the feedback for this response is {feedback2}. Please refine this response, considering the feedback and questions."
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": proposer_prompt3
            }
        ],
        model=MODEL_NAME
    )
    time.sleep(2)
    
    final_answer = chat_completion.choices[0].message.content
    data.loc[count, "two_agents_probing"] = final_answer
    # print(chat_completion.choices[0].message.content)
    time.sleep(2)
    count += 1
    
if not os.path.exists("./experimental_data"):
    os.mkdir("./experimental_data")

data.to_csv("./experimental_data/truthful_QA.csv")