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
    g++ \
    libc6-dev \
    libstdc++-12-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY feynman_engine/resources/form/form-5.0.0.tar.gz ./

RUN mkdir -p src \
    && tar -xzf form-5.0.0.tar.gz -C src \
    && cd src/form-* \
    && ./configure --disable-float --disable-parform \
    && make -j1 \
    && cp sources/form /build/form \
    && chmod +x /build/form

# ─── LoopTools builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS looptools-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    libstdc++-12-dev \
    gfortran \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY feynman_engine/resources/looptools/LoopTools-2.16.tar ./

RUN set -ex \
    && mkdir -p src \
    && tar -xf LoopTools-2.16.tar -C src \
    && cd src/LoopTools-2.16 \
    && export FFLAGS="-fPIC -O2" \
    && export CFLAGS="-fPIC -O2" \
    && export CC=gcc \
    && ./configure --prefix=/build/install \
    && make -j1 \
    && ls -la build/libooptools.a \
    && gfortran -shared \
         -o /build/liblooptools.so \
         -Wl,--whole-archive,$(pwd)/build/libooptools.a,--no-whole-archive \
         -lgfortran -lm

# ─── LHAPDF builder ──────────────────────────────────────────────────────────
# LHAPDF is the standard PDF library (Buckley et al., EPJ C 75 (2015) 132).
# We compile the C++ library + Python bindings against the same Python that
# the production image uses (3.11) and install to /opt/lhapdf so it's
# auto-discovered by feynman_engine.amplitudes.pdf at runtime.
FROM python:3.11-slim AS lhapdf-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    libc6-dev \
    libstdc++-12-dev \
    make \
    curl \
    ca-certificates \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY feynman_engine/resources/lhapdf/LHAPDF-6.5.5.tar.gz ./

RUN set -ex \
    && mkdir -p src \
    && tar -xzf LHAPDF-6.5.5.tar.gz -C src \
    && cd src/LHAPDF-6.5.5 \
    && PYTHON=$(which python) ./configure --prefix=/opt/lhapdf \
    && make -j2 \
    && make install

# Bundle a default PDF set (CT18LO).  We try to download it during build;
# if the build host lacks internet, the stage still produces a usable
# LHAPDF install and users can run `feynman install-pdf-set CT18LO` later.
RUN mkdir -p /opt/lhapdf/share/LHAPDF \
    && cd /opt/lhapdf/share/LHAPDF \
    && (curl -fsL "http://lhapdfsets.web.cern.ch/lhapdfsets/current/CT18LO.tar.gz" -o CT18LO.tar.gz \
        && tar xzf CT18LO.tar.gz && rm CT18LO.tar.gz \
        && echo "CT18LO bundled" \
        || echo "WARNING: CT18LO download failed during build; install via feynman install-pdf-set CT18LO")

# ─── OpenLoops builder ───────────────────────────────────────────────────────
# OpenLoops 2 (Buccioni et al., EPJ C 79 (2019) 866, arXiv:1907.13071) is the
# automated tree + one-loop amplitude generator used by FeynmanEngine for
# generic NLO over arbitrary processes.  Build the framework + bundle a
# minimal process library (ppllj = Drell-Yan + jet) so the production image
# can compute generic NLO out of the box.
FROM python:3.11-slim AS openloops-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY feynman_engine/resources/openloops/OpenLoops-OpenLoops-2.1.4.tar.gz ./

RUN set -ex \
    && mkdir -p src \
    && tar -xzf OpenLoops-OpenLoops-2.1.4.tar.gz -C src \
    && mv src/OpenLoops-OpenLoops-2.1.4 src/openloops \
    && cd src/openloops \
    && PYTHON=$(which python) ./scons \
    && mkdir -p /opt/openloops \
    && cp -r openloops openloops.cfg.tmpl scons SConstruct config examples \
            lib pyol lib_src include scons-local /opt/openloops/ \
    && chmod +x /opt/openloops/openloops /opt/openloops/scons \
    && mkdir -p /opt/openloops/proclib

# Bundle a curated process pack covering the major LHC analyses:
#   ppllj — Drell-Yan + jet
#   pptt  — top pair (NLO QCD)
#   pph   — gluon-fusion Higgs (loop-induced)
# Each ~50-100 MB compiled.  Other libraries (pphtt, ppvv, pphjj, pphh)
# are user-installed on demand via `feynman install-process <name>`.
RUN cd /opt/openloops \
    && for proc in ppllj pptt pph; do \
        (./openloops libinstall "$proc" \
          && echo "$proc bundled") \
          || echo "WARNING: $proc download failed during build; install via feynman install-process $proc"; \
       done

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

# Drop in the LHAPDF install + default PDF set from the builder stage.
# feynman_engine.amplitudes.pdf auto-discovers /opt/lhapdf and configures
# sys.path + LHAPDF_DATA_PATH + LD_LIBRARY_PATH at module import — no env
# vars needed.  Also probe /tmp/lhapdf-install (legacy/dev location).
COPY --from=lhapdf-builder /opt/lhapdf /opt/lhapdf

# Drop in the OpenLoops install + default ppllj process library from the
# builder stage.  feynman_engine.amplitudes.openloops_bridge auto-discovers
# /opt/openloops at import.  We need libgfortran5 (already installed above)
# for the Fortran shared libraries to load.
COPY --from=openloops-builder /opt/openloops /opt/openloops

ENV LHAPDF_DATA_PATH=/opt/lhapdf/share/LHAPDF \
    LD_LIBRARY_PATH=/opt/lhapdf/lib:/opt/openloops/lib:/opt/openloops/proclib \
    PYTHONPATH=/opt/lhapdf/lib/python3.11/site-packages:/opt/openloops/pyol/tools \
    OPENLOOPS_PREFIX=/opt/openloops

EXPOSE ${PORT:-10000}

CMD ["sh", "-c", "uvicorn feynman_engine.api.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
