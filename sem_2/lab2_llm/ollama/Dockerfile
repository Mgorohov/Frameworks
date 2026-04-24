FROM ollama/ollama:latest
COPY pull_model.sh /pull_model.sh
RUN chmod +x /pull_model.sh
ENTRYPOINT ["/pull_model.sh"]