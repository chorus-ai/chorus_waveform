FROM alpine/git as builder
WORKDIR /
RUN git clone https://github.com/chorus-ai/chorus_waveform.git

FROM eclipse-temurin:11.0.23_9-jre

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        libsndfile1
RUN mkdir /app
COPY --from=builder /chorus_waveform /app/chorus_waveform
RUN rm -rf /app/chorus_waveform/UVAFormatConverter
RUN mv /app/chorus_waveform/data /data
WORKDIR /app/chorus_waveform
COPY pyproject.toml .
RUN pip install pip==23.2.1 \
                setuptools==68.1.2 \
                flake8 \
                pytest
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps -e .
