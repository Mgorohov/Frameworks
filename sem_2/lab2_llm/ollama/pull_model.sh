#!/bin/bash
ollama serve &
sleep 3
ollama pull qwen2.5:0.5b
tail -f /dev/null