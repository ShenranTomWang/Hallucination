import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_collection_utils import generate, get_class

MODEL = os.getenv("MODEL")
START_IDX = int(os.getenv("START_IDX", 0))
END_IDX = int(os.getenv("END_IDX", -1))
DATASET = os.getenv("DATASET")
PROBE = int(os.getenv("PROBE", 0)) == 1
OUTPUT_FILE = f"./output_{DATASET}" + ("_probe" if PROBE else "") + ".txt"
PromptClass = get_class("templates", "Probing" if PROBE else "NonProbing")
DatasetClass = get_class(f"data.dataset", DATASET)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on device {device}")

data = DatasetClass()

count = START_IDX
end = END_IDX if END_IDX != -1 else data.len()
chat_completions = []
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=device, torch_dtype=torch.float16)
with open(OUTPUT_FILE, "a") as f:
    f.write("model loaded\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL, device_map=device, torch_dtype=torch.float16)
with open(OUTPUT_FILE, "a") as f:
    f.write("tokenizer loaded\n")
while count < end:
    try:
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{count}\n")
        question = data.get_question(count)
        answer1 = generate(model, tokenizer, question, device)
        # print(answer1 + "\n" + "*" * 100)
        data.write("single_agent", count, answer1)
        refiner_prompt1 = PromptClass.REFINER_PROMPT_1_TEMPLATE.substitute(
            question=question, answer1=answer1
        )
        
        feedback1 = generate(model, tokenizer, refiner_prompt1, device)
        data.write("feedback1", count, feedback1)
        # print(feedback1 + "\n" + "*" * 100)
        proposer_prompt2 = PromptClass.PROPOSER_PROMPT_2_TEMPLATE.substitute(
            question=question, answer1=answer1, feedback1=feedback1
        )
        # print(proposer_prompt2 + "\n" + "*" * 150)
        
        answer2 = generate(model, tokenizer, proposer_prompt2, device)
        # print(answer2 + "\n" + "*" * 100)
        data.write("two_agents", count, answer2)
        
        summarizer_prompt = PromptClass.SUMMARIZER_PROMPT_TEMPLATE.substitute(
            question=question, answer1=answer1, feedback1=feedback1, answer2=answer2
        )
        final_answer = generate(model, tokenizer, summarizer_prompt, device)
        # allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        # print(f"Allocated {allocated_memory} GB cuda memory")
        # print(final_answer + "\n" + "*" * 100)
        data.write("three_agents", count, final_answer)
    except Exception as e:
        with open(OUTPUT_FILE, "a") as f:
            f.write(e + "\n")
    finally:
        count += 1
        if not os.path.exists("./experimental_data"):
            os.mkdir("./experimental_data")
        data.to_csv(f"./experimental_data/{DATASET}" + ("_probe" if PROBE else "") + ".csv")
    # print("\n\n\n")
    
if not os.path.exists("./experimental_data"):
    os.mkdir("./experimental_data")

data.to_csv(f"./experimental_data/{DATASET}" + ("_probe" if PROBE else "") + ".csv")