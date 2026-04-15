FROM python:3.11-slim AS qgraf-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY qgraf-3.6.10.tgz ./

RUN mkdir -p src out \
    && tar -xzf qgraf-3.6.10.tgz -C src \
    && gfortran -O2 -o out/qgraf src/qgraf-3.6.10.f08 \
    && chmod +x out/qgraf

# ─── FORM builder ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS form-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY feynman_engine/resources/form/form-5.0.0.tar.gz ./

RUN mkdir -p src \
    && tar -xzf form-5.0.0.tar.gz -C src \
    && cd src/form-* \
    && ./configure --disable-float --disable-parform \
    && make -j4 \
    && cp sources/form /build/form \
    && chmod +x /build/form

# ─── LoopTools builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS looptools-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY feynman_engine/resources/looptools/LoopTools-2.16.tar ./

RUN mkdir -p src \
    && tar -xf LoopTools-2.16.tar -C src \
    && cd src \
    && ./configure --prefix=/build/install FFLAGS="-fPIC -O2" CFLAGS="-fPIC -O2" \
    && make -j4 \
    && gfortran -shared \
         -o /build/liblooptools.so \
         -Wl,--whole-archive build/libooptools.a --no-whole-archive \
         -lgfortran -lm

# ─── Production image ────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# LaTeX + SVG rendering stack.
# texlive-science ships tikz-feynman on Debian 12 (Bookworm).
# texlive-plain-generic provides many .sty files required by standalone.
# texlive-fonts-recommended prevents missing font warnings that abort lualatex.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pdf2svg \
    texlive-luatex \
    texlive-pictures \
    texlive-latex-extra \
    texlive-science \
    texlive-fonts-recommended \
    texlive-plain-generic \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying app code so this layer is cached
# across code-only changes.
COPY pyproject.toml README.md ./
COPY feynman_engine/ ./feynman_engine/
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Copy remaining sources (frontend, contrib, tests, etc.)
COPY . .

# Drop in the Linux QGRAF binary compiled in the builder stage.
# (The bin/ directory may contain a macOS binary from development — overwrite it.)
COPY --from=qgraf-builder /build/out/qgraf ./bin/qgraf
RUN chmod +x ./bin/qgraf \
    && { [ -f ./bin/qgraf_pipe ] && chmod +x ./bin/qgraf_pipe || true; }

# Drop in the Linux LoopTools shared library compiled in the builder stage.
COPY --from=looptools-builder /build/liblooptools.so ./bin/liblooptools.so

# Drop in the Linux FORM binary compiled in the builder stage.
COPY --from=form-builder /build/form ./bin/form
RUN chmod +x ./bin/form

EXPOSE ${PORT:-10000}

CMD ["sh", "-c", "uvicorn feynman_engine.api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
