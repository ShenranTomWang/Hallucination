from string import Template

class Probing():
    REFINER_PROMPT_1_TEMPLATE = Template("$question. Here is a response to this question: $answer1. Please provide feedback on this answer, being specific to the question and providing actionable feedback. Ask 2 questions regarding the aspects of the response that seem incorrect.")
    PROPOSER_PROMPT_2_TEMPLATE = Template("$question Here is an initial response $answer1 and a feedback $feedback1. Please refine this response, considering the feedback and questions.")
    SUMMARIZER_PROMPT_TEMPLATE = Template("$question Here is an initial response to this question: $answer1 Here is the initial feedback: $feedback1. Here is the second answer based on the feedback and doubts: $answer2. Please look at the chat history and provide a final response to $question based on it.")
    REFINER_PROMPT_2_TEMPLATE = Template("$question. Here is the initial response: $answer1, initial feedback: $feedback1, and the refined answer: $answer2. Please provide feedback on this answer, being specific to the question and providing actionable feedback. Ask 2 questions regarding the aspects of the response that seem incorrect.")
    PROPOSER_PROMPT_3_TEMPLATE = Template("$question Here is an initial response $answer1 and an initial feedback $feedback1. The response is then refined to $answer2, the feedback for this response is $feedback2. Please refine this response, considering the feedback and questions.")

class NonProbing():
    REFINER_PROMPT_1_TEMPLATE = Template("$question. Here is a response to this question: $answer1. Please provide feedback on this answer, being specific to the question and providing actionable feedback.")
    PROPOSER_PROMPT_2_TEMPLATE = Template("$question Here is an initial response $answer1 and a feedback $feedback1. Please refine this response, considering the feedback.")
    SUMMARIZER_PROMPT_TEMPLATE = Template("$question Here is an initial response to this question: $answer1 Here is the initial feedback: $feedback1. Here is the second answer based on the feedback: $answer2. Please look at the chat history and provide a final response to $question based on it.")
    REFINER_PROMPT_2_TEMPLATE = Template("$question. Here is the initial response: $answer1, initial feedback: $feedback1, and the refined answer: $answer2. Please provide feedback on this answer, being specific to the question and providing actionable feedback.")
    PROPOSER_PROMPT_3_TEMPLATE = Template("$question Here is an initial response $answer1 and an initial feedback $feedback1. The response is then refined to $answer2, the feedback for this response is $feedback2. Please refine this response, considering the feedback.")
