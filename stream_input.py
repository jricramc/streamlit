import streamlit as st
import openai
import numpy as np
import time
from dotenv import load_dotenv
import os

# Load API key from environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

openai.api_key = api_key

# Function Definitions
def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=answer_context,
            n=1
        )
    except Exception as e:
        st.warning("Retrying due to an error: {}".format(e))
        time.sleep(20)
        return generate_answer(answer_context)
    return completion

def construct_message(agents, question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}
    prefix_string = "These are the recent/updated opinions from other agents: "
    for agent in agents:
        agent_response = agent[-1]["content"]
        response = f"\n\n One agent response: ```{agent_response}```"
        prefix_string = prefix_string + response
    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    # This function now just returns the sentence, assuming the whole response is the answer.
    return sentence.strip()

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    return num

def run_debate(question, agents, rounds):
    agent_contexts = [[{"role": "user", "content": question}] for _ in range(agents)]
    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                message = construct_message(agent_contexts_other, question)
                agent_context.append(message)
            completion = generate_answer(agent_context)
            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)
    
    text_answers = []
    for agent_context in agent_contexts:
        text_answer = agent_context[-1]['content']
        text_answers.append(parse_answer(text_answer))
    
    if not text_answers:
        return "No valid responses from agents."

    try:
        return most_frequent(text_answers)
    except Exception as e:
        return f"Error in processing answers: {e}"

def run_zero_shot(question):
    completion = generate_answer([{"role": "user", "content": question}])
    zero_shot_answer = completion["choices"][0]["message"]["content"]
    return zero_shot_answer

# Streamlit UI
def run():
    st.title("Input Question for Multi-Agent Debate")

    question = st.text_area("Enter your question:")
    agents = st.slider("Number of Agents", min_value=1, max_value=10, value=2)
    rounds = st.slider("Number of Rounds", min_value=1, max_value=10, value=3)

    if st.button("Run Debate"):
        if not question:
            st.warning("Please enter a question.")
        else:
            st.write("Running multi-agent debate...")
            debate_answer = run_debate(question, agents, rounds)
            st.write("Multi-Agent Debate Answer:", debate_answer)

            st.write("Running zero-shot...")
            zero_shot_answer = run_zero_shot(question)
            st.write("Zero-Shot Answer:", zero_shot_answer)

