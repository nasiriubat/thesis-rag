# syntax=docker/dockerfile:1

FROM python:slim AS base


# Install dependencies only when needed
FROM base AS deps
WORKDIR /app
# Install dependencies based on the requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt


FROM deps AS runner
WORKDIR /app

RUN mkdir /data
VOLUME /data

EXPOSE 8080
ENV PORT=8080
ENV FLASK_ENV=production
ENV DATA_PATH=sqlite:////data
ENV BEHIND_PROXY=True

COPY . .

CMD ["./start.sh"]
