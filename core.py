import aiml
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import internal.kb_methods as kb
from internal.tfidf_module import TfidfManager
from internal.brain_tumor_mask import predict_mask
from internal.fuzzy_fever import assess_fever_description

# Глобальные переменные
kern = None
kb_expressions = None
tfidf_manager = None

def init_once():
    """Инициализация бота: AIML, TF-IDF, база знаний"""
    global kern, kb_expressions, tfidf_manager
    if kern:
        return  # Уже инициализировано

    kern = aiml.Kernel()
    kern.setTextEncoding(None)
    kern.bootstrap(learnFiles="internal/storage/healthbot.aiml")

    qna_df = pd.read_csv("internal/storage/healthcare_qna.csv")
    tfidf_manager = TfidfManager(qna_df['question'], qna_df['answer'])

    # Гарантировать, что файл базы существует
    kb.ensure_storage_path()
    kb_expressions = kb.load_kb()

def process_query(text: str, is_voice=False) -> dict:
    """Обработка запроса пользователя"""
    init_once()
    answer = kern.respond(text)

    if answer.startswith("#"):
        handle_result = handle_aiml_command(answer, text, is_voice)
        return {"answer": handle_result, "continue": False if handle_result == "Goodbye!" else True}
    else:
        if not answer.strip() or answer in ["I don't know how to respond to that", "I have no answer for that"]:
            fallback = tfidf_manager.get_most_similar_answer(text)
            return {"answer": fallback or "Извините, я не знаю, как на это ответить.", "continue": True}
        return {"answer": answer, "continue": True}

def fetch_wikipedia_section(query, section):
    """Получение секции из Википедии"""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": query,
        "format": "json",
        "prop": "sections"
    }

    section_fallbacks = {
        "Overview":  ["Overview", "Introduction", "Summary", "General"],
        "Symptoms":  ["Symptoms", "Signs and symptoms"],
        "Causes":    ["Causes",   "Risk factors"],
        "Treatment": ["Treatment","Management", "Therapy"]
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        section_number = None
        fallbacks = section_fallbacks.get(section, [section])

        for fallback in fallbacks:
            for s in data.get("parse", {}).get("sections", []):
                if fallback.lower() in s["line"].lower():
                    section_number = s["index"]
                    break
            if section_number:
                break

        if not section_number:
            sections_list = data.get("parse", {}).get("sections", [])
            if sections_list:
                section_number = sections_list[0].get("index")

        if section_number:
            params["prop"] = "text"
            params["section"] = section_number
            section_response = requests.get(url, params=params)
            if section_response.status_code == 200:
                section_data = section_response.json()
                section_html = section_data.get("parse", {}).get("text", {}).get("*", "")
                soup = BeautifulSoup(section_html, "html.parser")
                section_text = soup.get_text(separator=" ").strip()

                section_text = re.sub(r"\[\d+\]", "", section_text)
                section_text = re.sub(r"\[edit\]", "", section_text)

                if section == "Symptoms":
                    response_text = f"Here are common symptoms of {query.capitalize()}:\n\n{section_text}"
                elif section == "Causes":
                    response_text = f"Main causes and risk factors for {query.capitalize()}:\n\n{section_text}"
                elif section == "Treatment":
                    response_text = f"Treatments and management strategies for {query.capitalize()}:\n\n{section_text}"
                else:
                    response_text = f"Overview of {query.capitalize()}:\n\n{section_text}"

                if len(response_text) > 1000:
                    last_period = response_text[:1000].rfind(".")
                    response_text = response_text[:last_period + 1] + "..." if last_period != -1 else response_text[:1000] + "..."
                return response_text
    return "I couldn't find that information on Wikipedia."

def handle_aiml_command(answer, user_input, is_voice=False):
    """Обработка команд AIML вида #1$param"""
    global kern, kb_expressions, tfidf_manager
    params = answer[1:].split("$")
    cmd = int(params[0])

    if cmd == 0:
        return params[1] if len(params) > 1 else "Goodbye!"

    elif cmd == 1:  # Wikipedia: Overview
        return fetch_wikipedia_section(params[1], "Overview")

    elif cmd == 2:  # Wikipedia: Symptoms
        return fetch_wikipedia_section(params[1], "Symptoms")

    elif cmd == 3:  # Wikipedia: Causes
        return fetch_wikipedia_section(params[1], "Causes")

    elif cmd == 4:  # Wikipedia: Treatment
        return fetch_wikipedia_section(params[1], "Treatment")

    elif cmd == 100:  # TF-IDF
        query = params[1]
        return tfidf_manager.get_most_similar_answer(query) or "I don't know the answer."

    elif cmd == 101:  # Add fact to KB
        return kb.add_fact(kb_expressions, params[1])

    elif cmd == 102:  # Check fact in KB
        return kb.check_fact(kb_expressions, params[1])

    elif cmd == 300:  # Fuzzy Fever
        return assess_fever_description(params[1])

    elif cmd == 301:  # Brain image classification (requires frontend to upload image separately)
        return "Image classification should be done via /brain/segment endpoint."

    return "Unknown command."
