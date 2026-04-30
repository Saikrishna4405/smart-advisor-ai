import google.generativeai as genai
import tensorflow as tf
import os

print("Gemini Library Version:", genai.__version__)
print("Tensorflow Version:", tf.__version__)

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialization: SUCCESS")
    except Exception as e:
        print("Gemini model initialization: FAILED", e)
else:
    print("GOOGLE_API_KEY not found in environment.")
