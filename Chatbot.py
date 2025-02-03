import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load intents from JSON file
with open("intents.json", "r") as file:
    data = json.load(file)

# Prepare data for training
patterns = []
tags = []
descriptions = {}

for category in data["categories"]:
    category_name = category["category"]
    
    for example in category["examples"]:
        patterns.append(example)
        tags.append(category_name)

    # Store category descriptions
    descriptions[category_name] = category.get("description", "No description available.")

# Vectorize the patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, tags)

# Chatbot function with WellIntent & FallbackIntent
def waste_chatbot(user_input):
    input_vec = vectorizer.transform([user_input])
    try:
        category_name = model.predict(input_vec)[0]
        
        # Check if it's a WellIntent
        if category_name == "WellIntent":
            return random.choice(["You're welcome!", "Glad to help!", "Anytime! 😊"])
        
        return descriptions.get(category_name, "No description available.")
    
    except:
        # Fallback Intent for unknown inputs
        return random.choice(["Sorry, I couldn't understand that.", "Can you rephrase?", "I'm not sure how to categorize this item."])

# Streamlit-based chat interface
st.title("Waste Segregation Chatbot")
st.write("Ask about how to dispose of a specific waste item.")

user_input = st.text_input("What item do you want to dispose of?", "")
if user_input:
    response = waste_chatbot(user_input.lower())
    st.write(f"Bot: {response}")
