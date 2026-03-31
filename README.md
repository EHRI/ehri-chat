# EHRI chat: RAG, Graph-RAG and MCP using the European Holocaust Research Infrastructure Knowledge Graph data

EHRI chat is a specialised Retrieval-Augmented Generation (RAG) system designed to enhance Large Language Model (LLM) responses using authoritative and curated data from the [European Holocaust Research Infrastructure Knowledge Graph (EHRI-KG)](https://lod.ehri-project-test.eu/). 
It exploits the idea of providing additional context to LLMs in order to improve their accuracy and surpass the training cut-off date.

## Features

- **Multi-Model Support:** Integration with Google Gemini, Mistral AI, and local models via an OpenAI chat completions compatible API.
- **Knowledge Graph Integration:** Direct interaction with the EHRI-KG SPARQL endpoint.
- **Different context retrieval methods:** Combines semantic vector search with structured graph queries, including RAG, GraphRAG and MCP (see [Implemented context techniques](#implemented-context-techniques)).
- **Flexible Interfaces:** Includes a Command Line Interface (CLI) and a Flask-based web server with a chat UI (see [Web server](#web-server)).
- **Automated Indexing:** Built-in tools for generating and managing FAISS vector indices.

## Prerequisites

- Python 3.10+
- API Keys for Gemini and/or Mistral (when using cloud models).
- A local LLM provider implementing an Open AI chat completion compatible API (e.g., llama.cpp, Ollama, LMStudio, vLLM, etc.). Default: Ollama

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ehriChat
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### API Keys
Set the following environment variables depending on the model you intend to use:
```bash
export MISTRAL_API_KEY="your_mistral_key"
export GEMINI_API_KEY="your_gemini_key"
```

### Local Models
The system expects a local LLM endpoint (like Ollama or llama.cpp). By default, it is configured to use Ollama under `localhost:11434` and the `qwen3-vl:2b-instruct-q4_K_M` model. However, you can easily change this configuration by tweaking the `server.py` and/or `main.py` files.

### FAISS Indices
Before running this tool using the RAG or GraphRAG modes for the first time, you need to generate the vector indices:
```bash
python -m ehri_graph_rag.embeddings.embeddings_manager
```
Indices will be stored under `conf/faiss/`.

## Usage

### Command Line Interface (CLI)
Query the system directly from the terminal:
```bash
python -m ehri_graph_rag.main --prompt "Tell me about archives in Poland" --mode GraphRAG --model gemini
```
**Modes:** `vanilla`, `RAG`, `GraphRAG`, `MCP`  
**Models:** `qwen`, `mistral`, `gemini`

### Web Server
Start the Flask server to provide a web interface. When deploying in production, consider using a production-ready server like `gunicorn`.
```bash
flask --app ehri_graph_rag.server run
```
- **Web UI:** Accessible at `http://localhost:5000/`

You will find a familiar interface in which you can comfortably interact with the system using all the aforementioned modes and models. The web is based on [Andrej Karpathy's nanochat HTML interface](https://github.com/karpathy/nanochat/blob/master/nanochat/ui.html) with some additions to support this specific case.

## Implemented context techniques

At the moment, the following techniques are implemented in the project:

- **Standard RAG:** Uses vector similarity search (FAISS) over chunked EHRI portal data. The chunking occurs over certain fields of the supported entities whereas other entities are ingested in a list-of-attributes style.
- **GraphRAG:** Only a small amount of data (name, type, URI and description) are ingested in a per-entity FAISS index. A heuristic approach is followed to retrieve the most relevant entities per entity type (using vector similarity search) and some linked entities.
- **Model Context Protocol (MCP):** Integrates with an EHRI MCP server that provides LLMs the ability to have real-time access to the EHRI-KG data. The implementation is based on an OpenAPI spec (see [EHRI-KG OpenAPI](https://lod.ehri-project-test.eu/api-local/)) developed using [grlc](https://github.com/CLARIAH/grlc) in which the methods tagged as `mcp` are available as an MCP server: `https://lod.ehri-project-test.eu/mcp`. One of the advantages of this method is that you can integrate it with compatible LLM providers (satisfactorily tested against: Mistral's Le Chat, Anthropic's Claude and Open WebUI).

The supported entities from the EHRI-KG are:
- ehri:Country
- ehri:Institution
- ehri:RecordSet

In the future more entities will be supported and additional retrieval methods (mainly for GraphRAG and MCP) will be developed.

## Analytics
This tool is a research project and therefore a small SQLite database is created when using it. You can use it to analyse the introduced prompts together with the generated responses in order to get a better understanding of how the combination of enrichement options and the different models affects the results.

The database will be automatically created in the first run and stored under `db/activity.db`

## Limitations
- Context flooding: Some of the context retrieval methods may incur in a higher consumption of the context limit. The application was designed to minimise this by only inputting the most relevant fields for each entity, trying in this way to reduce redundant data. However, for some specific cases the context can be flooded (especially when using small local LLMs) or the application can have a higher than expected token consumption.
- No chat history: The chat history is not passed to the model and therefore only the last message is used in the generation. This is enforced in order to avoid flooding the context as explained before and to speed up the generation. However, you can easily change this behaviour in the `server.py` file.
