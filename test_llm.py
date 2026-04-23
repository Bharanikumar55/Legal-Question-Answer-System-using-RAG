import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content("What is Artificial Intelligence?")

    print("\n✅ LLM WORKING\n")
    print(response.text)

except Exception as e:
    print("\n❌ ERROR\n", e)