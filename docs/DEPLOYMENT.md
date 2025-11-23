# Deployment Guide

This guide covers installation, configuration, and deployment of the Vector Database MCP Server in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Systemd Service](#systemd-service)
- [MCP Integration](#mcp-integration)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### Required Software

1. **Python 3.11 or higher**
   ```bash
   python --version  # Should show 3.11.0 or higher
   ```

2. **uv package manager**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or with pip
   pip install uv

   # Verify installation
   uv --version
   ```

3. **Ollama**
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Windows
   # Download from https://ollama.ai/download

   # Verify installation
   ollama --version
   ```

4. **Embedding Model**
   ```bash
   # Pull the required embedding model (336MB)
   ollama pull mxbai-embed-large

   # Verify model is available
   ollama list | grep mxbai-embed-large
   ```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk Space | 2 GB | 10+ GB |
| OS | macOS 11+, Ubuntu 20.04+, Windows 10+ | Latest stable |

## Installation Methods

### Method 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/vector-mcp-server.git
cd vector-mcp-server

# Create virtual environment and install dependencies
uv sync

# Verify installation
uv run python -m vector_mcp --help
```

### Method 2: pip Installation (Coming Soon)

```bash
# Install from PyPI (when published)
pip install vector-mcp

# Verify installation
python -m vector_mcp --help
```

### Method 3: Docker (See Docker Deployment section)

```bash
docker pull yourusername/vector-mcp:latest
```

## Configuration

### Environment Variables

Create a `.env` file or export environment variables:

```bash
# Vector Database Configuration
export VECTOR_MCP_DB_PATH="$HOME/.vector_mcp/chroma_db"

# Ollama Configuration
export VECTOR_MCP_OLLAMA_HOST="http://localhost:11434"
export VECTOR_MCP_EMBEDDING_MODEL="mxbai-embed-large"

# Chunking Configuration
export VECTOR_MCP_CHUNK_SIZE=512
export VECTOR_MCP_CHUNK_OVERLAP=50

# Query Configuration
export VECTOR_MCP_MAX_RESULTS=5
export VECTOR_MCP_MAX_TOKENS=3000

# Logging Configuration
export VECTOR_MCP_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
export VECTOR_MCP_LOG_FILE="$HOME/.vector_mcp/logs/server.log"
```

### Configuration File (Optional)

Create `~/.vector_mcp/config.yaml`:

```yaml
database:
  path: ~/.vector_mcp/chroma_db

ollama:
  host: http://localhost:11434
  model: mxbai-embed-large
  timeout: 30
  batch_size: 32

chunking:
  chunk_size: 512
  chunk_overlap: 50

query:
  max_results: 5
  max_tokens: 3000

logging:
  level: INFO
  file: ~/.vector_mcp/logs/server.log
```

Load config file:
```bash
uv run python -m vector_mcp --config ~/.vector_mcp/config.yaml
```

### Directory Structure

The server creates this directory structure automatically:

```
~/.vector_mcp/
├── chroma_db/           # ChromaDB persistent storage
│   ├── chroma.sqlite3   # Metadata database
│   └── [collection_id]/ # Vector embeddings
├── logs/
│   └── server.log       # Application logs
└── config.yaml          # Optional configuration
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Create `docker-compose.yml`**:

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  vector-mcp:
    image: yourusername/vector-mcp:latest
    container_name: vector-mcp
    depends_on:
      - ollama
    environment:
      - VECTOR_MCP_DB_PATH=/data/chroma_db
      - VECTOR_MCP_OLLAMA_HOST=http://ollama:11434
      - VECTOR_MCP_EMBEDDING_MODEL=mxbai-embed-large
    volumes:
      - vector_data:/data
      - ./documents:/documents:ro  # Mount your documents
    restart: unless-stopped
    command: ["python", "-m", "vector_mcp"]

volumes:
  ollama_data:
  vector_data:
```

2. **Start services**:

```bash
# Start all services
docker-compose up -d

# Pull embedding model (first time only)
docker exec ollama ollama pull mxbai-embed-large

# View logs
docker-compose logs -f vector-mcp

# Stop services
docker-compose down
```

### Using Dockerfile Only

1. **Build image**:

```bash
docker build -t vector-mcp:latest .
```

2. **Run container**:

```bash
# Start Ollama first
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest

# Pull embedding model
docker exec ollama ollama pull mxbai-embed-large

# Start vector-mcp
docker run -d \
  --name vector-mcp \
  --link ollama:ollama \
  -e VECTOR_MCP_OLLAMA_HOST=http://ollama:11434 \
  -v vector_data:/data \
  -v $(pwd)/documents:/documents:ro \
  vector-mcp:latest
```

### Multi-stage Dockerfile

The included `Dockerfile` uses multi-stage builds for optimization:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/
ENV PATH="/app/.venv/bin:$PATH"
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "-m", "vector_mcp"]
```

## Systemd Service

For Linux servers, run as a systemd service for automatic startup and restart.

### Service File

Create `/etc/systemd/system/vector-mcp.service`:

```ini
[Unit]
Description=Vector Database MCP Server
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=vector-mcp
Group=vector-mcp
WorkingDirectory=/opt/vector-mcp
Environment="PATH=/opt/vector-mcp/.venv/bin:/usr/bin"
Environment="VECTOR_MCP_DB_PATH=/var/lib/vector-mcp/chroma_db"
Environment="VECTOR_MCP_OLLAMA_HOST=http://localhost:11434"
Environment="VECTOR_MCP_LOG_FILE=/var/log/vector-mcp/server.log"
ExecStart=/opt/vector-mcp/.venv/bin/python -m vector_mcp
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vector-mcp

[Install]
WantedBy=multi-user.target
```

### Installation Steps

1. **Create service user**:
```bash
sudo useradd -r -s /bin/false -d /opt/vector-mcp vector-mcp
```

2. **Install application**:
```bash
sudo mkdir -p /opt/vector-mcp
sudo chown vector-mcp:vector-mcp /opt/vector-mcp
cd /opt/vector-mcp
sudo -u vector-mcp git clone https://github.com/yourusername/vector-mcp.git .
sudo -u vector-mcp uv sync
```

3. **Create data directories**:
```bash
sudo mkdir -p /var/lib/vector-mcp/chroma_db
sudo mkdir -p /var/log/vector-mcp
sudo chown -R vector-mcp:vector-mcp /var/lib/vector-mcp
sudo chown -R vector-mcp:vector-mcp /var/log/vector-mcp
```

4. **Enable and start service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vector-mcp
sudo systemctl start vector-mcp
sudo systemctl status vector-mcp
```

5. **View logs**:
```bash
sudo journalctl -u vector-mcp -f
```

## MCP Integration

### Claude Code Configuration

Add to `~/.claude/config/mcp.json`:

```json
{
  "mcpServers": {
    "vector-db": {
      "command": "uv",
      "args": ["run", "python", "-m", "vector_mcp"],
      "env": {
        "VECTOR_MCP_DB_PATH": "/Users/harrison/.vector_mcp/chroma_db",
        "VECTOR_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### For Docker Deployment

```json
{
  "mcpServers": {
    "vector-db": {
      "command": "docker",
      "args": [
        "exec",
        "vector-mcp",
        "python",
        "-m",
        "vector_mcp"
      ]
    }
  }
}
```

### Testing MCP Connection

From Claude Code:
```
> List available MCP tools
> Call index_file with a test document
```

Or use the MCP inspector:
```bash
# Install MCP inspector
npm install -g @modelcontextprotocol/inspector

# Test connection
mcp-inspector uv run python -m vector_mcp
```

## Troubleshooting

### Ollama Not Starting

**Symptoms**: `Connection refused: http://localhost:11434`

**Solutions**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Or as a background service (Linux)
sudo systemctl start ollama

# macOS (if installed via Homebrew)
brew services start ollama

# Check Ollama logs
journalctl -u ollama -f  # Linux
tail -f ~/.ollama/logs/server.log  # macOS
```

### Model Not Found

**Symptoms**: `Error: model 'mxbai-embed-large' not found`

**Solutions**:
```bash
# Pull the model
ollama pull mxbai-embed-large

# Verify model
ollama list

# Test model
ollama run mxbai-embed-large "test"
```

### ChromaDB Permission Errors

**Symptoms**: `Permission denied: /path/to/chroma_db`

**Solutions**:
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/.vector_mcp/chroma_db

# Fix permissions
chmod 755 ~/.vector_mcp
chmod 755 ~/.vector_mcp/chroma_db

# Or specify different path
export VECTOR_MCP_DB_PATH=/tmp/chroma_db
```

### High Memory Usage

**Symptoms**: Python process using >4GB RAM

**Solutions**:

1. **Reduce batch size**:
   ```bash
   export VECTOR_MCP_EMBEDDING_BATCH_SIZE=16  # Default: 32
   ```

2. **Reduce chunk size**:
   ```bash
   export VECTOR_MCP_CHUNK_SIZE=256  # Default: 512
   ```

3. **Process files individually**:
   ```python
   # Instead of index_directory, use multiple index_file calls
   ```

### Slow Embedding Performance

**Symptoms**: Indexing takes >2s per chunk

**Solutions**:

1. **Enable GPU acceleration** (if available):
   ```bash
   # Verify GPU support
   ollama list

   # GPU should be automatically detected
   ```

2. **Increase batch size**:
   ```bash
   export VECTOR_MCP_EMBEDDING_BATCH_SIZE=64
   ```

3. **Use smaller model** (lower quality):
   ```bash
   ollama pull all-minilm
   export VECTOR_MCP_EMBEDDING_MODEL=all-minilm
   ```

### MCP Connection Issues

**Symptoms**: Claude Code can't connect to MCP server

**Solutions**:

1. **Verify MCP config syntax**:
   ```bash
   cat ~/.claude/config/mcp.json | jq .
   ```

2. **Test command manually**:
   ```bash
   uv run python -m vector_mcp --help
   ```

3. **Check Claude Code logs**:
   ```bash
   tail -f ~/.claude/logs/mcp.log
   ```

4. **Restart Claude Code**

## Maintenance

### Database Backup

```bash
# Stop the server
sudo systemctl stop vector-mcp  # Linux
# or kill the process

# Backup ChromaDB
tar -czf vector-mcp-backup-$(date +%Y%m%d).tar.gz ~/.vector_mcp/chroma_db

# Restart server
sudo systemctl start vector-mcp
```

### Database Cleanup

Remove outdated or unwanted documents:

```python
# Via Python
from vector_mcp.storage.chroma_store import ChromaStore
from pathlib import Path

store = ChromaStore(Path("~/.vector_mcp/chroma_db").expanduser())

# Delete by source file
results = store.search(
    query_embedding=[0]*1024,  # dummy
    where={"source_file": "/path/to/old/doc.md"}
)
store.delete([r["id"] for r in results])
```

### Log Rotation

Create `/etc/logrotate.d/vector-mcp`:

```
/var/log/vector-mcp/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 vector-mcp vector-mcp
    sharedscripts
    postrotate
        systemctl reload vector-mcp > /dev/null 2>&1 || true
    endscript
}
```

### Monitoring

**Check service status**:
```bash
sudo systemctl status vector-mcp
```

**Monitor resource usage**:
```bash
# CPU and memory
top -p $(pgrep -f vector_mcp)

# Disk usage
du -sh ~/.vector_mcp/chroma_db
```

**Query performance metrics**:
```bash
# Check logs for query latency
grep "search_time" ~/.vector_mcp/logs/server.log
```

### Updates

```bash
# Pull latest code
cd /opt/vector-mcp
sudo -u vector-mcp git pull

# Update dependencies
sudo -u vector-mcp uv sync

# Restart service
sudo systemctl restart vector-mcp
```

## Security Considerations

### File Permissions

```bash
# Restrict database access
chmod 700 ~/.vector_mcp/chroma_db

# Restrict config file
chmod 600 ~/.vector_mcp/config.yaml
```

### Network Security

- Ollama binds to localhost by default (secure)
- For remote access, use SSH tunneling:
  ```bash
  ssh -L 11434:localhost:11434 user@remote-host
  ```

### Data Privacy

- All data stays local (no external API calls)
- ChromaDB stores embeddings locally
- No telemetry or external connections

## Performance Tuning

### For Large Datasets (>10,000 docs)

1. **Increase batch size**:
   ```bash
   export VECTOR_MCP_EMBEDDING_BATCH_SIZE=128
   ```

2. **Use SSD for ChromaDB**:
   ```bash
   export VECTOR_MCP_DB_PATH=/path/to/ssd/chroma_db
   ```

3. **Allocate more RAM to Docker** (if using Docker):
   ```yaml
   # docker-compose.yml
   services:
     vector-mcp:
       mem_limit: 8g
   ```

### For High Query Volume

1. **Cache query embeddings**:
   - Implement query cache (future feature)

2. **Use faster model**:
   ```bash
   ollama pull all-minilm  # Smaller, faster model
   ```

3. **Optimize chunk size**:
   - Smaller chunks = faster search but less context
   - Larger chunks = slower search but more context

## Production Checklist

- [ ] Ollama installed and running
- [ ] Embedding model pulled (mxbai-embed-large)
- [ ] Environment variables configured
- [ ] Database directory created with correct permissions
- [ ] Log rotation configured
- [ ] Systemd service enabled (Linux)
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] MCP integration tested with Claude Code
- [ ] Resource limits set (if using Docker)

## Support

For additional help:
- GitHub Issues: https://github.com/yourusername/vector-mcp/issues
- Documentation: https://github.com/yourusername/vector-mcp/wiki
- Discord: https://discord.gg/vector-mcp
