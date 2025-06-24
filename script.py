import google.generativeai as genai

genai.configure(api_key="AIzaSyCQOaC4E7HPWq_c3uuJmFZzcaF_FE6O5vU")

models = genai.list_models()

for model in models:
    print(model)
