FROM python:3.11-slim

WORKDIR /app

# Install forge globally inside the container
COPY . .
RUN pip install --no-cache-dir -e .

# Default working directory is /workspace (mounted by user)
WORKDIR /workspace

# Forge config inside container reads from environment variables
# Users pass: docker run -e GEMINI_API_KEY=... -e LOCAL_MODEL=...
ENV LM_STUDIO_BASE_URL=http://host.docker.internal:1234
ENV MASTER_MODEL=gemini-2.5-flash
ENV LOCAL_CTX_SIZE=8192

ENTRYPOINT ["forge"]
CMD ["--help"]
