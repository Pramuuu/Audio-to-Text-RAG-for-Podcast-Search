# 🚀 Deployment Guide - Advanced Podcast RAG Search

This guide provides comprehensive instructions for deploying the Advanced Podcast RAG Search system in various environments.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space
- **GPU**: Optional but recommended for faster processing (CUDA 11.0+)

### Software Dependencies
- **Python**: Latest stable version
- **FFmpeg**: For audio processing
- **Git**: For version control
- **Docker**: For containerized deployment (optional)

## 🏗️ Installation Methods

### Method 1: Direct Installation (Recommended for Development)

#### 1. Clone Repository
```bash
git clone <repository-url>
cd RAG_MODEL_PODCAST_SEARCH
```

#### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional dependencies
pip install -e .
```

#### 4. Download Required Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download WhisperX models (will be downloaded automatically on first use)
```

#### 5. Environment Configuration
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env file with your API keys
nano .env
```

### Method 2: Docker Deployment (Recommended for Production)

#### 1. Build Docker Image
```bash
# Build the image
docker build -t podcast-rag-search .

# Or use docker-compose
docker-compose build
```

#### 2. Run Container
```bash
# Run with docker
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e COHERE_API_KEY=your_key \
  podcast-rag-search

# Or use docker-compose
docker-compose up -d
```

### Method 3: Cloud Deployment

#### Deploy to Heroku
```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login

# Create app
heroku create your-podcast-rag-app

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key
heroku config:set COHERE_API_KEY=your_key

# Deploy
git push heroku main
```

#### Deploy to Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/podcast-rag-search

# Deploy to Cloud Run
gcloud run deploy podcast-rag-search \
  --image gcr.io/PROJECT_ID/podcast-rag-search \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔧 Configuration

### Environment Variables

#### Required API Keys
```bash
# OpenAI API Key (for GPT-4 and embeddings)
OPENAI_API_KEY=sk-your-openai-key-here

# Cohere API Key (for alternative embeddings)
COHERE_API_KEY=your-cohere-key-here

# HuggingFace Token (for speaker diarization)
HUGGINGFACE_TOKEN=your-hf-token-here
```

#### Audio Processing Configuration
```bash
# Audio quality settings
AUDIO_SAMPLE_RATE=16000
WHISPERX_MODEL=large-v2  # Options: tiny, base, small, medium, large, large-v2
WHISPERX_DEVICE=cuda     # Options: cpu, cuda
```

#### RAG System Configuration
```bash
# Retrieval settings
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7
RERANKING_ENABLED=true
```

### Configuration Files

#### Streamlit Configuration
Create `~/.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

## 🚀 Running the Application

### Development Mode
```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Or run demo
python demo.py
```

### Production Mode
```bash
# Using Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Using Gunicorn (if using custom WSGI)
gunicorn -w 4 -b 0.0.0.0:8501 streamlit_app:app
```

### Background Service (Linux/macOS)
```bash
# Create systemd service file
sudo nano /etc/systemd/system/podcast-rag.service

# Service content:
[Unit]
Description=Podcast RAG Search Application
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/RAG_MODEL_PODCAST_SEARCH
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/streamlit run streamlit_app.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable podcast-rag
sudo systemctl start podcast-rag
```

## 📊 Monitoring and Logging

### Application Logs
```bash
# View Streamlit logs
tail -f ~/.streamlit/logs/streamlit.log

# View system logs (if using systemd)
sudo journalctl -u podcast-rag -f
```

### Performance Monitoring
```bash
# Monitor system resources
htop
nvidia-smi  # If using GPU

# Monitor application metrics
# The app includes built-in analytics dashboard
```

## 🔒 Security Considerations

### API Key Security
- Never commit API keys to version control
- Use environment variables or secure key management
- Rotate keys regularly
- Use least-privilege access

### Network Security
```bash
# Firewall configuration (Ubuntu)
sudo ufw allow 8501
sudo ufw enable

# Reverse proxy with Nginx
sudo apt install nginx
sudo nano /etc/nginx/sites-available/podcast-rag
```

#### Nginx Configuration Example
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/TLS Configuration
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com
```

## 📈 Scaling and Performance

### Horizontal Scaling
```bash
# Load balancer configuration
# Use multiple instances behind a load balancer

# Docker Swarm
docker swarm init
docker stack deploy -c docker-compose.yml podcast-rag

# Kubernetes
kubectl apply -f k8s-deployment.yaml
```

### Performance Optimization
```bash
# GPU acceleration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🧪 Testing and Validation

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/ -v --cov=src

# Run specific tests
pytest tests/test_audio_processor.py -v
```

### System Validation
```bash
# Validate configuration
python config.py

# Run demo
python demo.py

# Check system status
streamlit run streamlit_app.py --server.headless true
```

## 🚨 Troubleshooting

### Common Issues

#### Audio Processing Errors
```bash
# Check FFmpeg installation
ffmpeg -version

# Install FFmpeg (Ubuntu)
sudo apt update
sudo apt install ffmpeg

# Install FFmpeg (macOS)
brew install ffmpeg
```

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues
```bash
# Reduce batch size
export WHISPERX_BATCH_SIZE=8

# Use CPU instead of GPU
export WHISPERX_DEVICE=cpu
```

### Log Analysis
```bash
# Check error logs
grep -i error ~/.streamlit/logs/streamlit.log

# Monitor system resources
top -p $(pgrep -f streamlit)
```

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [WhisperX Documentation](https://github.com/m-bain/whisperX)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)

### Support
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)
- Community Forum: [Discord/Slack channels]
- Documentation: [Project Wiki]

## 🎯 Next Steps

After successful deployment:

1. **Upload Sample Podcasts**: Test the system with sample audio files
2. **Configure Monitoring**: Set up alerts and monitoring dashboards
3. **Performance Tuning**: Optimize based on your specific use case
4. **Security Audit**: Review and enhance security measures
5. **Backup Strategy**: Implement regular backups and disaster recovery
6. **User Training**: Train users on system capabilities and best practices

---

**Happy Deploying! 🚀**

For additional support, please refer to the project documentation or create an issue in the repository. 