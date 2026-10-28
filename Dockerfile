# UNTESTED: Docker was not available in the build sandbox. Build and smoke-test before publishing.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 ragdoc
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV RAGDOC_BASE_DIR=/data
VOLUME /data
USER ragdoc
# stdio transport: run with `docker run -i --rm -v "$PWD":/data ragdoc-mcp`
ENTRYPOINT ["ragdoc-mcp"]
