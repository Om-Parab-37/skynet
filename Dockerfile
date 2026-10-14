# One image serves both the API and the Temporal worker; the command selects the
# role. Keeping them identical means a capability can never be present in one
# and missing from the other.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so editing source does not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src/skynet/__init__.py src/skynet/__init__.py
RUN uv sync --no-dev --extra all --no-install-project

COPY src/ src/
COPY prompts/ prompts/
COPY agents.yaml ./
RUN uv sync --no-dev --extra all

EXPOSE 8000
CMD ["uvicorn", "skynet.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
