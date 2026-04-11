FROM python:3.11-slim AS qgraf-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gfortran \
    tar \
    && rm -rf /var/lib/apt/lists/*

COPY qgraf-3.6.10.tgz /build/qgraf-3.6.10.tgz

RUN mkdir -p /build/src /build/out \
    && tar -xzf /build/qgraf-3.6.10.tgz -C /build/src \
    && gfortran -O2 -o /build/out/qgraf /build/src/qgraf-3.6.10.f08 \
    && chmod +x /build/out/qgraf

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# System packages needed for SVG/PDF rendering.
# QGRAF itself is not bundled here because its source is distributed separately.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    pdf2svg \
    texlive-luatex \
    texlive-pictures \
    texlive-latex-extra \
    texlive-science \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && pip install .

COPY --from=qgraf-builder /build/out/qgraf /app/bin/qgraf

RUN chmod +x /app/bin/qgraf
RUN if [ -f /app/bin/qgraf_pipe ]; then chmod +x /app/bin/qgraf_pipe; fi

EXPOSE 10000

CMD ["sh", "-c", "uvicorn feynman_engine.api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
