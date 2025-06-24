import requests

video_id = "8JuWdXrCmWg"
url = f"https://www.youvideototext.com/api/video/{video_id}?lang=auto"

response = requests.get(url)
if response.status_code == 200:
    print(response.json())
else:
    print("Failed to fetch transcript")
