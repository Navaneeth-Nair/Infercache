# ==============================================================================
# Multi-stage Dockerfile for InferCache: Ultra-Low-RAM Semantic Caching Proxy
# ==============================================================================
FROM rust:1.82-slim-bookworm AS builder

WORKDIR /app

# Install build essentials for C dependencies (OpenSSL / Ring / Tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src tests && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release || true && \
    rm -rf src tests

# Copy full application source
COPY . .

# Build production binary
RUN cargo build --release

# ==============================================================================
# Minimal Runtime Container (~35MB RAM footprint)
# ==============================================================================
FROM debian:bookworm-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 infercache
USER infercache

COPY --from=builder /app/target/release/infercache /app/infercache

ENV HOST=0.0.0.0 \
    PORT=8080 \
    RUST_LOG=infercache=info,tower_http=info

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/infercache"]
