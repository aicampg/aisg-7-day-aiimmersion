import logging
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

def set_logging():
    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the minimum logging level for the logger

    # Create a stream handler that outputs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set the handler's level to INFO or DEBUG

    # Create a formatter to customize log output (optional, but recommended)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    # Add the stream handler to the logger
    logger.addHandler(stream_handler)



    #set up the global logger to have the same level and handler
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler])

set_logging()
# 1. Load your data
logging.info("Loading documents...")
documents = SimpleDirectoryReader(input_dir="ollama-documents/").load_data()
logging.info(f"Loaded {len(documents)} documents.")
# 2. Setup Ollama LLM
# we want the embeddings to synthesize document info!
logging.info("Setting up Ollama LLM...")
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
logging.info("Setting up Ollama Embedding...")

ollama_embedding = OllamaEmbedding(
    model_name="llama3.2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0, "request_timeout": 120},
)

# 3. Create the index
logging.info("Creating index...")
index = VectorStoreIndex.from_documents(documents, embed_model=ollama_embedding)
logging.info("Setting up index completed...")
# 4. Query the index
logging.info("Querying index...")
query_engine = index.as_query_engine()

sitrep = query_engine.query("Latest villian info?")
print(sitrep)
