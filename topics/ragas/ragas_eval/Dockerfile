FROM johndellenbaugh/rag-comparison:latest

COPY requirements.txt /tmp/ragas-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ragas-requirements.txt

# Stub out langchain_community.chat_models.vertexai — removed in 0.3.x but
# still imported by ragas internals. Physical file is more reliable than
# sys.modules patching.
RUN python -c "\
import os, sys; \
p = next((os.path.join(d, 'langchain_community', 'chat_models') for d in sys.path if os.path.isdir(os.path.join(d, 'langchain_community', 'chat_models'))), None); \
p and open(os.path.join(p, 'vertexai.py'), 'w').write('class ChatVertexAI: pass\n') and print('stub created at', p)"

# Disable NumExpr SIMD — this VM CPU has no AVX512 and numexpr crashes silently
ENV NUMEXPR_DISABLE=1

COPY . /app/ragas_eval/
WORKDIR /app
EXPOSE 7861

CMD ["python", "ragas_eval/main.py"]
