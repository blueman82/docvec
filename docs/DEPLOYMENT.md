# Deployment Guide

This guide covers installation, configuration, and deployment of the DocVec in various environments.

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
git clone https://github.com/yourusername/docvec.git
cd docvec

# Create virtual environment and install dependencies
uv sync

# Verify installation
uv run python -m docvec --help
```

### Method 2: pip Installation (Coming Soon)

```bash
# Install from PyPI (when published)
pip install docvec

# Verify installation
python -m docvec --help
```

### Method 3: Docker (See Docker Deployment section)

```bash
docker pull yourusername/docvec:latest
```

## Configuration

### Environment Variables

Configuration can be set via CLI arguments or environment variables with the `DOCVEC_` prefix:

```bash
# Vector Database Configuration
export DOCVEC_DB_PATH="./chroma_db"

# Ollama Configuration
export DOCVEC_HOST="http://localhost:11434"
export DOCVEC_MODEL="nomic-embed-text"
export DOCVEC_TIMEOUT=30

# Chunking Configuration
export DOCVEC_CHUNK_SIZE=256
export DOCVEC_BATCH_SIZE=16

# Collection Configuration
export DOCVEC_COLLECTION="documents"

# Logging Configuration
export DOCVEC_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

Or use CLI arguments:
```bash
python -m docvec \
  --db-path ./chroma_db \
  --host http://localhost:11434 \
  --model nomic-embed-text \
  --chunk-size 256 \
  --batch-size 16 \
  --collection documents \
  --log-level INFO
```

### Quick Start with Environment Variables

```bash
# Set environment variables
export DOCVEC_DB_PATH="./chroma_db"
export DOCVEC_HOST="http://localhost:11434"
export DOCVEC_MODEL="nomic-embed-text"

# Run server
uv run python -m docvec
```

Alternatively, pass all arguments on command line:
```bash
uv run python -m docvec \
  --db-path ./chroma_db \
  --host http://localhost:11434 \
  --model nomic-embed-text \
  --log-level DEBUG
```

### Directory Structure

The server uses the specified data directory (default: `./chroma_db`):

```
./chroma_db/
├── chroma.sqlite3       # Metadata database
└── [collection_id]/     # Vector embeddings
```

Or when using `DOCVEC_DB_PATH`:
```
$DOCVEC_DB_PATH/
├── chroma.sqlite3       # Metadata database
└── [collection_id]/     # Vector embeddings
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

  docvec:
    image: yourusername/docvec:latest
    container_name: docvec
    depends_on:
      - ollama
    environment:
      - DOCVEC_DB_PATH=/data/chroma_db
      - DOCVEC_HOST=http://ollama:11434
      - DOCVEC_MODEL=nomic-embed-text
      - DOCVEC_LOG_LEVEL=INFO
    volumes:
      - vector_data:/data
      - ./documents:/documents:ro  # Mount your documents
    restart: unless-stopped
    command: ["python", "-m", "docvec"]

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
docker-compose logs -f docvec

# Stop services
docker-compose down
```

### Using Dockerfile Only

1. **Build image**:

```bash
docker build -t docvec:latest .
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
docker exec ollama ollama pull nomic-embed-text

# Start docvec
docker run -d \
  --name docvec \
  --link ollama:ollama \
  -e DOCVEC_HOST=http://ollama:11434 \
  -e DOCVEC_MODEL=nomic-embed-text \
  -v vector_data:/data \
  -v $(pwd)/documents:/documents:ro \
  docvec:latest
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
CMD ["python", "-m", "docvec"]
```

## Systemd Service

For Linux servers, run as a systemd service for automatic startup and restart.

### Service File

Create `/etc/systemd/system/docvec.service`:

```ini
[Unit]
Description=DocVec - Document Vector Database Server
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=docvec
Group=docvec
WorkingDirectory=/opt/docvec
Environment="PATH=/opt/docvec/.venv/bin:/usr/bin"
Environment="DOCVEC_DB_PATH=/var/lib/docvec/chroma_db"
Environment="DOCVEC_HOST=http://localhost:11434"
Environment="DOCVEC_MODEL=nomic-embed-text"
Environment="DOCVEC_LOG_LEVEL=INFO"
ExecStart=/opt/docvec/.venv/bin/python -m docvec
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=docvec

[Install]
WantedBy=multi-user.target
```

### Installation Steps

1. **Create service user**:
```bash
sudo useradd -r -s /bin/false -d /opt/docvec docvec
```

2. **Install application**:
```bash
sudo mkdir -p /opt/docvec
sudo chown docvec:docvec /opt/docvec
cd /opt/docvec
sudo -u docvec git clone https://github.com/yourusername/docvec.git .
sudo -u docvec uv sync
```

3. **Create data directories**:
```bash
sudo mkdir -p /var/lib/docvec/chroma_db
sudo chown -R docvec:docvec /var/lib/docvec
```

4. **Enable and start service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable docvec
sudo systemctl start docvec
sudo systemctl status docvec
```

5. **View logs**:
```bash
sudo journalctl -u docvec -f
```

## MCP Integration

### Claude Code Configuration

Add to `~/.claude/config/mcp.json`:

```json
{
  "mcpServers": {
    "docvec": {
      "command": "uv",
      "args": ["run", "python", "-m", "docvec"],
      "env": {
        "DOCVEC_DB_PATH": "/Users/harrison/.docvec/chroma_db",
        "DOCVEC_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### For Docker Deployment

```json
{
  "mcpServers": {
    "docvec": {
      "command": "docker",
      "args": [
        "exec",
        "docvec",
        "python",
        "-m",
        "docvec"
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
mcp-inspector uv run python -m docvec
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
sudo chown -R $USER:$USER ./chroma_db

# Fix permissions
chmod 755 ./chroma_db

# Or specify different path
export DOCVEC_DB_PATH=/tmp/chroma_db
# Or use CLI argument
python -m docvec --db-path /tmp/chroma_db
```

### High Memory Usage

**Symptoms**: Python process using >4GB RAM

**Solutions**:

1. **Reduce batch size**:
   ```bash
   export DOCVEC_BATCH_SIZE=8
   # Or via CLI
   python -m docvec --batch-size 8
   ```

2. **Reduce chunk size**:
   ```bash
   export DOCVEC_CHUNK_SIZE=128
   # Or via CLI
   python -m docvec --chunk-size 128
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
   export DOCVEC_BATCH_SIZE=32
   # Or via CLI
   python -m docvec --batch-size 32
   ```

3. **Use smaller model** (lower quality):
   ```bash
   ollama pull all-minilm
   export DOCVEC_MODEL=all-minilm
   # Or via CLI
   python -m docvec --model all-minilm
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
   uv run python -m docvec --help
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
sudo systemctl stop docvec  # Linux
# or kill the process

# Backup ChromaDB
tar -czf docvec-backup-$(date +%Y%m%d).tar.gz ~/.docvec/chroma_db

# Restart server
sudo systemctl start docvec
```

### Database Cleanup

Remove outdated or unwanted documents:

```python
# Via Python
from docvec.storage.chroma_store import ChromaStore
from pathlib import Path

store = ChromaStore(Path("~/.docvec/chroma_db").expanduser())

# Delete by source file
results = store.search(
    query_embedding=[0]*1024,  # dummy
    where={"source_file": "/path/to/old/doc.md"}
)
store.delete([r["id"] for r in results])
```

### Log Rotation

Create `/etc/logrotate.d/docvec`:

```
/var/log/docvec/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 docvec docvec
    sharedscripts
    postrotate
        systemctl reload docvec > /dev/null 2>&1 || true
    endscript
}
```

### Monitoring

**Check service status**:
```bash
sudo systemctl status docvec
```

**Monitor resource usage**:
```bash
# CPU and memory
top -p $(pgrep -f docvec)

# Disk usage
du -sh ~/.docvec/chroma_db
```

**Query performance metrics**:
```bash
# Check logs for query latency
grep "search_time" ~/.docvec/logs/server.log
```

### Updates

```bash
# Pull latest code
cd /opt/docvec
sudo -u docvec git pull

# Update dependencies
sudo -u docvec uv sync

# Restart service
sudo systemctl restart docvec
```

## Security Considerations

### File Permissions

```bash
# Restrict database access
chmod 700 ~/.docvec/chroma_db

# Restrict config file
chmod 600 ~/.docvec/config.yaml
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
   export DOCVEC_BATCH_SIZE=32
   # Or via CLI
   python -m docvec --batch-size 32
   ```

2. **Use SSD for ChromaDB**:
   ```bash
   export DOCVEC_DB_PATH=/path/to/ssd/chroma_db
   # Or via CLI
   python -m docvec --db-path /path/to/ssd/chroma_db
   ```

3. **Allocate more RAM to Docker** (if using Docker):
   ```yaml
   # docker-compose.yml
   services:
     docvec:
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
- GitHub Issues: https://github.com/yourusername/docvec/issues
- Documentation: https://github.com/yourusername/docvec/wiki
- Discord: https://discord.gg/docvec
