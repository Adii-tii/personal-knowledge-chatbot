from modules.ingestion.webpage_parser import extract_text_from_url

url = "https://www.ibm.com/think/ai-agents"
text = extract_text_from_url(url)

if text:
    print("✅ Webpage parsed successfully!")
    print(text[:1000])  # Print first 1000 chars
else:
    print("❌ Failed to parse webpage.")
