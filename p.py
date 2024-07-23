import ollama
import chromadb
import fitz  # PyMuPDF

# Load PDF and extract text
pdf_path = 'C:\\Users\\m\\Desktop\\1\\1.pdf'
pdf_document = fitz.open(pdf_path)
documents = [page.get_text() for page in pdf_document]

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

prompt = "Summarize the book"

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
