import google.generativeai as genai

genai.configure(api_key="API_KEY_HERE")

models = genai.list_models()

for model in models:
    print(model)
