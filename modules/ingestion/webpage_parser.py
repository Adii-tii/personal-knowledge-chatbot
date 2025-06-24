from newspaper import Article

def extract_text_from_url(url: str) -> str:
    """
    Extracts readable text from a news/blog article using newspaper3k.
    Args:
        url (str): The URL of the web article.

    Returns:
        str: Cleaned article text.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        print(f"[Error] Failed to extract from URL: {e}")
        return ""
