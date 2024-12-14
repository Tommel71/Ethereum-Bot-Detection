# Use a Python base image
FROM ghcr.io/astral-sh/uv:bookworm-slim

# Install system dependencies, including Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    clang \
    libpq-dev \
    texlive-latex-base \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-extra \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \

# Set working directory
WORKDIR /app

# Copy only the files needed for uv to minimize cache invalidation
COPY pyproject.toml uv.lock ./

# Sync dependencies in a virtual environment
RUN uv python install 3.10
RUN uv sync

# Copy the rest of the application
COPY . .

CMD ["/bin/bash", "-c", "uv run pipeline.py"]

