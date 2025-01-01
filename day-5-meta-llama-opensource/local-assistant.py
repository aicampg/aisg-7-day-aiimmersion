from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 1. Load your data
documents = SimpleDirectoryReader(input_dir="ollama-documents/").load_data()

# 2. Setup Ollama LLM
# we want the embeddings to synthesize document info!
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

ollama_embedding = OllamaEmbedding(
    model_name="llama3.2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0, "request_timeout": 120},
)

# 3. Create the index
index = VectorStoreIndex.from_documents(documents, embed_model=ollama_embedding)

# 4. Query the index
query_engine = index.as_query_engine()
sitrep = query_engine.query("Latest villian info?")
print(sitrep)
