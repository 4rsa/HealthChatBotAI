import os
from nltk.sem import Expression

# Абсолютный путь: <папка-модуля>/storage/kb_file.kb
BASE_DIR = os.path.dirname(__file__)           # internal/
KB_FILE  = os.path.join(BASE_DIR, "storage", "kb_file.kb")


# Function to load the KB from the file
def load_kb():
    try:
        with open(KB_FILE, "r") as file:
            return [Expression.fromstring(line.strip()) for line in file if line.strip()]
    except FileNotFoundError:
        # If the file doesn't exist, return an empty KB
        return []
    
def ensure_storage_path():
    """Создаёт папку и файл хранения KB, если их нет"""
    folder = os.path.dirname(KB_FILE)
    os.makedirs(folder, exist_ok=True)
    if not os.path.isfile(KB_FILE):
        open(KB_FILE, "w").close()

# Function to save the KB to the file
def save_kb(kb_expressions):
    with open(KB_FILE, "w") as file:
        for expr in kb_expressions:
            file.write(str(expr) + "\n")

def preprocess_query(query):
    """
    Converts certain natural language queries to FOL syntax.
    - "sneezing is a symptom of cold" -> "Symptom(Sneezing, Cold)"
    - "X is good" -> "Good(X)"
    - "X is bad" -> "Bad(X)"
    """
    query = query.strip().lower()  # Standardize to lowercase for parsing

    # 1) Handle " is good"
    if " is good" in query:
        # e.g., "apple is good" -> Good(Apple)
        subject = query.replace(" is good", "").strip()
        return f"Good({subject.capitalize()})"

    # 2) Handle " is bad"
    if " is bad" in query:
        # e.g., "pizza is bad" -> Bad(Pizza)
        subject = query.replace(" is bad", "").strip()
        return f"Bad({subject.capitalize()})"

    # 3) Handle "is a symptom of" pattern
    if "is a symptom of" in query:
        parts = query.split(" is a symptom of ")
        if len(parts) == 2:
            subject, predicate = parts
            return f"Symptom({subject.capitalize().strip()}, {predicate.capitalize().strip()})"

    # 4) Handle generic " is " pattern
    if " is " in query:
        parts = query.split(" is ")
        if len(parts) == 2:
            subject, predicate = parts
            return f"Symptom({subject.capitalize().strip()}, {predicate.capitalize().strip()})"

    return query  # Return as-is if no rule applied

def add_fact(kb_expressions, fact):
    """
    Adds a new fact to the KB after preprocessing.
    When "X is good" or "X is bad" is already contradicted 
    or duplicated by existing knowledge, it will not store the new one.
    """
    # Preprocess the input to convert it to valid FOL syntax
    processed_fact = preprocess_query(fact)

    try:
        new_fact = Expression.fromstring(processed_fact)

        # If the new fact is Good(X), check for conflicts or redundancy
        if str(new_fact).startswith("Good("):
            # Example: Good(Apple)
            subject = str(new_fact)[5:-1]  # everything after Good( and before )
            
            # Build "Good(X)" & "Bad(X)" for comparison
            good_expr = Expression.fromstring(f"Good({subject})")
            bad_expr = Expression.fromstring(f"Bad({subject})")

            if good_expr in kb_expressions:
                return f"I already know that {subject} is good."
            if bad_expr in kb_expressions:
                return f"I know that {subject} is bad, so I won't remember it as good."
            
            # Otherwise add the new fact
            kb_expressions.append(good_expr)
            save_kb(kb_expressions)
            return f"OK, I will remember that {subject} is good."

        # If the new fact is Bad(X), check for conflicts or redundancy
        if str(new_fact).startswith("Bad("):
            subject = str(new_fact)[4:-1]
            
            good_expr = Expression.fromstring(f"Good({subject})")
            bad_expr = Expression.fromstring(f"Bad({subject})")

            if bad_expr in kb_expressions:
                return f"I already know that {subject} is bad."
            if good_expr in kb_expressions:
                return f"I know that {subject} is good, so I won't remember it as bad."

            # Otherwise add the new fact
            kb_expressions.append(bad_expr)
            save_kb(kb_expressions)
            return f"OK, I will remember that {subject} is bad."

        # For all other facts (e.g., Symptom(...))
        if new_fact in kb_expressions:
            return f"I already know that {processed_fact}."
        else:
            kb_expressions.append(new_fact)
            save_kb(kb_expressions)
            return f"OK, I will remember that {processed_fact}."

    except Exception as e:
        return f"Error adding fact: {e}"

def check_fact(kb_expressions, query):
    """
    Checks if the given query is in the KB, applying FOL syntax if necessary.
    Returns "Correct" if the fact exists, "Incorrect" if it does not,
    and "I don't know" if the query cannot be evaluated.
    """
    try:
        # Preprocess the query to FOL syntax
        processed_query = preprocess_query(query)
        query_expr = Expression.fromstring(processed_query)

        # Check if the query is in the KB
        if query_expr in kb_expressions:
            return "Correct"
        else:
            return "Incorrect"
    except Exception:
        # Handle evaluation errors gracefully
        return "I don't know"
