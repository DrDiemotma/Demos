

import httpx
import streamlit as st


HOST: str = "http://localhost"
PORT: int = 39891

def chat_message(prompt_text: str, chat_history: list[dict[str, str]] | None = None, max_output_tokens: int = 150, temperature: float = 0.2, timeout: float = 60.0):
    if chat_history is None:
        chat_history = [{
            "role": "system",
            "content": "You are a requirement engineering assistant to make sure everything is using IREB standards. "
            "You answer in full sentences and every sentence has one 'shall' or 'must'."
        }]
    chat_history.append({
        "role": "user",
        "content": prompt_text
    })

    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]) + "\nassistant:"
    payload = {
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{HOST}:{PORT}/v1/completions", json=payload)

    if response.status_code == 200:
        resp_json = response.json()
        result_text = resp_json.get('choices')[0].get('text')
        chat_history.append({"role": "assistant", "content": result_text})
        return result_text, chat_history
    else:
        print("ERROR:", response.status_code, response.text)
        return "", chat_history

if "requirement_history" not in st.session_state:
    requirement_history = [{
        "role": "system",
        "content": "You are reviewing requirements. " 
                   "Make sure every answer is a full sentence and contains one shall or must. "
                   "Answer in as few sentences as possible."
    }]
    st.session_state["requirement_history"] = requirement_history
else:
    requirement_history = st.session_state["requirement_history"]
if "test_history" not in st.session_state:
    test_history = [{
        "role": "system",
        "content": "You are a system to evaluate tests for requirements. "
                   "Each test must be falsifiable, clear, short. "
                   "Your answer must include acceptance criteria for the test. "
    }]
    st.session_state["test_history"] = test_history
else:
    test_history = st.session_state["test_history"]

input_column, output_column = st.columns(2)
user_input = input_column.text_area("Requirement input:", height=300)
if input_column.button("Submit requirement") and len(user_input) > 10:
    chat_message(user_input, chat_history=requirement_history)

requirement_answers = [x for x in requirement_history if x["role"] == "assistant"]
if len(requirement_answers) > 0:
    result_requirement = requirement_answers[-1]["content"]
    output_column.markdown(f"**REQUIREMENT:**\n\n{result_requirement}")

input_test, output_test = st.columns(2)
user_test_input = input_test.text_area("Test input:", height=300)
if input_test.button("Submit test") and len(user_test_input) > 10:
    chat_message(user_test_input, chat_history=test_history)

test_answers = [x for x in test_history if x["role"] == "assistant"]
if len(test_answers) > 0:
    result_test = test_answers[-1]["content"]
    output_test.markdown(f"**TEST:**\n\n{result_test}")

