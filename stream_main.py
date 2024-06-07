import streamlit as st
import openai
import numpy as np
import time
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load API key from environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

openai.api_key = api_key

# Function Definitions
def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []
    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue
        bullet = bullet[idx:]
        if len(bullet) != 0:
            bullets.append(bullet)
    return bullets

def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo-0301",
                  messages=answer_context,
                  n=1)
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
        response = "\n\n One agent response: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")
    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    return num

def run_simulation(agents, rounds, evaluation_rounds):
    scores = []
    generated_description = {}
    for _ in tqdm(range(evaluation_rounds)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)
        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for _ in range(agents)]
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)
        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt)
                    agent_context.append(message)
                completion = generate_answer(agent_context)
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
        text_answers = []
        for agent_context in agent_contexts:
            text_answer = agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)
            if text_answer is None:
                continue
            text_answers.append(text_answer)
        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)
        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue
    return np.mean(scores), np.std(scores) / (len(scores) ** 0.5)

def zero_shot_simulation(evaluation_rounds):
    scores = []
    for _ in tqdm(range(evaluation_rounds)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)
        answer = a + b * c + d - e * f
        question = "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.".format(a, b, c, d, e, f)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=[{"role": "user", "content": question}],
            n=1)
        response = completion["choices"][0]["message"]["content"]
        response = response.replace(",", ".")
        predicted_answer = parse_answer(response)
        if predicted_answer == answer:
            scores.append(1)
        else:
            scores.append(0)
    return np.mean(scores), np.std(scores) / (len(scores) ** 0.5)

# Streamlit UI
def run ():
    st.title("Multi-Agent Debate vs. Zero-Shot Comparison")


    st.write("""
    This application compares the performance of a multi-agent debate system with a zero-shot learning system. 
    Both systems are tasked with solving a specific mathematical equation: a + b * c + d - e * f, where a, b, c, d, e, and f are random integers between 0 and 30.
    The agents' answers are evaluated for correctness. 

    The multi-agent system involves multiple agents debating and refining their answers over several rounds. 
    Each agent independently generates an answer, then they share their answers and refine them based on the other agents' responses.

    The zero-shot system, on the other hand, provides an answer without any prior specific training on the task. 
    It generates an answer based solely on the question and does not have the opportunity to refine its answer based on other responses.
    """)
    
    agents = st.slider("Number of Agents", min_value=1, max_value=10, value=2)
    rounds = st.slider("Number of Rounds", min_value=1, max_value=10, value=3)
    evaluation_rounds = st.slider("Number of Evaluation Rounds", min_value=10, max_value=100, value=20)

    if st.button("Run Simulation"):
        st.write("Running multi-agent simulation...")
        multi_agent_mean, multi_agent_std = run_simulation(agents, rounds, evaluation_rounds)
        st.write("Multi-Agent Simulation - Mean Score:", multi_agent_mean)
        st.write("Multi-Agent Simulation - Standard Deviation:", multi_agent_std)

        st.write("Running zero-shot simulation...")
        zero_shot_mean, zero_shot_std = zero_shot_simulation(evaluation_rounds)
        st.write("Zero-Shot Simulation - Mean Score:", zero_shot_mean)
        st.write("Zero-Shot Simulation - Standard Deviation:", zero_shot_std)
