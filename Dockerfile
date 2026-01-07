FROM python:3.11-slim-bookworm

# Install system dependencies required for OpenCV
# libgl1-mesa-glx may be needed even for headless in some versions, but usually headless is fine.
# We definitely need libglib2.0-0 for cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run application
# We use `uv run` to ensure we use the environment
CMD ["uv", "run", "app.py"]
