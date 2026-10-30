# Transformer Studio

A small Streamlit demonstration of Transformer-based text generation using
`Qwen/Qwen2.5-1.5B-Instruct`.

## Run locally

The existing project virtual environment already contains the required packages.

```bash
source venv/bin/activate
streamlit run app.py
```

The first generation downloads the model from Hugging Face. Subsequent
generations reuse the cached model. On a MacBook Air with Apple Silicon, the
app uses MPS when available and falls back to the CPU otherwise.

## Features

- Creative story generation
- Natural passage continuation without repeating the input
- Temperature and maximum-token controls
- Cached model loading and visible generation errors
