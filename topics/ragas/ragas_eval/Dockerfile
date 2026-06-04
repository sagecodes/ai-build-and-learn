FROM johndellenbaugh/rag-comparison:latest

COPY requirements.txt /tmp/ragas-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ragas-requirements.txt

COPY . /app/ragas_eval/
WORKDIR /app
EXPOSE 7861

CMD ["python", "ragas_eval/main.py"]
