# Self-Hosted LLM Setup with Ollama

## Quick Start

### 1. Install Ollama
- **macOS**: Download from https://ollama.ai/download/mac
- **Linux**: `curl https://ollama.ai/install.sh | sh`
- **Windows**: Download from https://ollama.ai/download/windows

### 2. Start Ollama Server
```bash
# macOS/Linux
ollama serve

# Windows: Just run the Ollama app (it starts automatically)
```

The API will be available at `http://localhost:11434`

### 3. Download a Model
```bash
# Download Mistral (7B, recommended for RAG)
ollama pull mistral

# Or choose another 3B-7B GPT-like model:
ollama pull neural-chat      # 7B, good for chat
ollama pull orca-mini:3b     # 3B, compact
ollama pull phi              # 2.7B, very fast
ollama pull dolphin-mixtral  # Smaller, good quality
```

Check installed models:
```bash
ollama list
```

## Configuration

Update `RAGConfig` in your code:

```python
config = RAGConfig(
    llm_model="mistral",           # Model name
    ollama_base_url="http://localhost:11434",  # Ollama API endpoint
    llm_temperature=0.1,           # Lower = more focused
)

rag = MiniRAGSystem(config)
```

## Available Models by Size

### 3B Models (Fastest)
- `orca-mini:3b` - Good quality, very fast
- `phi` - Microsoft's 2.7B, excellent quality

### 7B Models (Balanced)
- `mistral` - **Recommended**, fast and capable
- `neural-chat` - Optimized for conversation
- `dolphin-mixtral` - High quality outputs

### Larger (More Capable, Slower)
- `llama2` - 7B, general purpose
- `llama2:13b` - 13B, more capable

## Model Performance Comparison

| Model | Size | Speed | Quality | VRAM |
|-------|------|-------|---------|------|
| phi | 2.7B | ⚡⚡⚡ | Good | 2-3GB |
| orca-mini:3b | 3B | ⚡⚡⚡ | Good | 3-4GB |
| mistral | 7B | ⚡⚡ | Excellent | 5-8GB |
| neural-chat | 7B | ⚡⚡ | Excellent | 5-8GB |
| llama2 | 7B | ⚡⚡ | Very Good | 5-8GB |

## Testing Ollama Connection

```python
from langchain.llms import Ollama

llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434"
)

# Test the connection
response = llm("What is Python?")
print(response)
```

## Troubleshooting

### "Connection refused"
- Make sure Ollama is running: `ollama serve`
- Check if port 11434 is accessible
- Verify `ollama_base_url` matches your setup

### "Model not found"
- Pull the model: `ollama pull mistral`
- Check available models: `ollama list`

### Out of Memory (OOM)
- Use a smaller model (phi or orca-mini:3b)
- Reduce chunk size in RAGConfig
- Close other applications

### Slow Responses
- You might have a CPU-only system
- Use GPU support: Install CUDA drivers for Ollama
- Try a smaller model (phi or orca-mini:3b)

## Resources

- Ollama Documentation: https://github.com/ollama/ollama
- Model Library: https://ollama.ai/library
- API Guide: https://github.com/ollama/ollama/blob/main/docs/api.md
