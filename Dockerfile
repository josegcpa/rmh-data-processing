FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.5.28 /uv /uvx /bin/

RUN mkdir -p /app
COPY pyproject.toml uv.lock .python-version /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN apt-get update && \
    apt-get install build-essential git -y && \
    apt-get clean
RUN uv sync --frozen

COPY entrypoint.sh dicom-to-nifti.sh /app
COPY scripts /app/scripts
COPY .env_docker /app/.env

ENV DICOM_OUT_DIR="/dicom_tmp"

ENTRYPOINT ["bash", "entrypoint.sh"]