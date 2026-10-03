# syntax=docker/dockerfile:1.7
# Self-contained eval image: harness + Copilot CLI + pinned ARC-AGI-3 games + pinned Frontier-CS
# polyomino files. Everything is downloaded at build time; a run needs no network except the
# model endpoint.
#
#   docker build -t ttc-arc --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#   docker run --rm -e OPENAI_BASE_URL=... -e OPENAI_API_KEY=... ttc-arc \
#       /app/run_trial --task arc --game lp85 --mode team --k 3 --trial 0 --results-dir /tmp/results --max-wall-seconds 43200
#   docker run --rm -e OPENAI_BASE_URL=... -e OPENAI_API_KEY=... ttc-arc \
#       /app/run_trial --task polyomino --mode team --k 3 --trial 0 --results-dir /tmp/results --max-wall-seconds 10800
#   docker run --rm -v $PWD/results:/r ttc-arc /app/verify_trial /r       # re-score collected results
#   docker run --rm ttc-arc /app/scripts/container_selftest.sh            # isolation + stub end-to-end

ARG NODE_IMAGE=node:22.23.3-bookworm-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

FROM ${NODE_IMAGE} AS node

FROM ${PYTHON_IMAGE}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

# Tools agents commonly use in their shell; util-linux provides setpriv, procps provides pkill;
# g++ is the polyomino judge's compiler (the official judge also uses g++ -std=gnu++17).
RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux procps ripgrep git ca-certificates libstdc++6 g++ make \
    && rm -rf /var/lib/apt/lists/*

# Node 22 from the pinned node image, and Copilot CLI from a lockfile with integrity hashes.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
COPY docker/copilot/package.json docker/copilot/package-lock.json /opt/copilot/
RUN cd /opt/copilot && npm ci --no-audit --no-fund \
    && ln -s /opt/copilot/node_modules/.bin/copilot /usr/local/bin/copilot \
    && npm cache clean --force \
    && COPILOT_OFFLINE=true COPILOT_AUTO_UPDATE=false copilot --version

# numpy for the agents' own analysis scripts (system python3, which agents get on PATH).
COPY agent-requirements.lock /tmp/agent-requirements.lock
RUN pip install --require-hashes -r /tmp/agent-requirements.lock && rm /tmp/agent-requirements.lock

# The harness, in its own venv with hash-locked dependencies.
WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN python -m venv /app/.venv && /app/.venv/bin/pip install --require-hashes -r /app/requirements.lock
COPY pyproject.toml README.md /app/
COPY ttc /app/ttc
RUN /app/.venv/bin/pip install --no-deps /app

# Pinned games, readable only by root (agents run as unprivileged users and cannot read them).
RUN /app/.venv/bin/ttc download --dir /opt/arc_envs \
    && chown -R root:root /opt/arc_envs && chmod -R go-rwx /opt/arc_envs \
    && /app/.venv/bin/ttc verify --dir /opt/arc_envs

# Pinned Frontier-CS polyomino files (statement, checker, testlib, 70 test cases), checker
# pre-built; root-only so neither agents nor the judged programs can read the test data.
RUN /app/.venv/bin/ttc fetch-frontiercs --dir /opt/frontiercs \
    && g++ /opt/frontiercs/algorithmic/problems/0/chk.cc -O2 -pipe -std=gnu++17 \
           -I /opt/frontiercs/algorithmic/judge/include -o /opt/frontiercs/chk \
    && chown -R root:root /opt/frontiercs && chmod -R go-rwx /opt/frontiercs \
    && /app/.venv/bin/ttc verify-frontiercs --dir /opt/frontiercs

# One unprivileged user per agent slot (teams up to k=8), and a separate user that runs the
# programs agents submit for judging (not in the agents' group).
RUN groupadd --system ttc-agents \
    && for i in 0 1 2 3 4 5 6 7; do \
         useradd --system --gid ttc-agents --no-create-home --home-dir /nonexistent \
                 --shell /usr/sbin/nologin "ttc-agent$i"; \
       done \
    && useradd --system --user-group --no-create-home --home-dir /nonexistent \
               --shell /usr/sbin/nologin ttc-judge

COPY scripts/run_trial scripts/verify_trial scripts/container_selftest.sh /app/scripts/
COPY tests/mock_llm.py /app/tests/mock_llm.py
COPY tests/fixtures /app/tests/fixtures
RUN ln -s /app/scripts/run_trial /app/run_trial && ln -s /app/scripts/verify_trial /app/verify_trial \
    && chmod 755 /app/scripts/*

ARG GIT_COMMIT=unknown
RUN printf '{"git_commit": "%s", "copilot_cli": "%s"}\n' "$GIT_COMMIT" \
        "$(COPILOT_OFFLINE=true copilot --version | head -1)" > /app/build_info.json

ENV COPILOT_AUTO_UPDATE=false COPILOT_OFFLINE=true
# No ENTRYPOINT/CMD on purpose: run /app/run_trial (or /app/verify_trial) explicitly.
