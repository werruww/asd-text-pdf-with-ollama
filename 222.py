import ollama
import chromadb
import requests
from bs4 import BeautifulSoup

# Function to fetch and parse text from a web page
def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text from paragraphs, headers, and other relevant HTML tags
    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
    text = ' '.join([para.get_text() for para in paragraphs])
    
    return text

# URL to fetch text from
url = 'https://ollama.com/blog/embedding-models'
document_text = fetch_text_from_url(url)

# Split the text into chunks or sections if necessary (for simplicity, we'll use the entire text)
documents = [document_text]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# Store each document in a vector embedding database
for i, d in enumerate(documents):
    response = ollama.embeddings(model="nomic-embed-text:latest", prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

prompt = "What are embedding models?"

response = ollama.embeddings(
    prompt=prompt,
    model="nomic-embed-text:latest"
)
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]

# Generate a response combining the prompt and the document retrieved in the previous step
output = ollama.generate(
    model="Mistral.gguf:latest",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])
