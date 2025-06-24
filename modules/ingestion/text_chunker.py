def chunk_text(text, max_chunk_size = 1500, overlap = 50):
    chunks = []
    start = 0
    text_len = len(text)

    while(start < text_len):
        end = start + max_chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks