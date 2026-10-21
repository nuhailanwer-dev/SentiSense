# SentiSense

https://github.com/user-attachments/assets/c44ca867-51d4-4a5b-8f7e-9541a77e56b3

## Streamlit interface (recommended)

Install the Python dependencies, then start the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit opens the app in your browser. The interface is in `streamlit_app.py`;
model loading and prediction are in `sentiment.py`. The model loads once and is
cached while the Streamlit app is running. The app graphs the overall positive,
neutral, and negative rates across analyzed entries in the current session.
Refreshing the page clears the graph and entry history.

## Terminal interface

Run the local sentiment model in interactive mode:

```bash
python test_distilbert.py
```

The model loads from `Model/` by default. Enter a message at the `Text>` prompt
to get its sentiment and confidence scores. Enter `q` to end the session.

For one-off or batch predictions, use `--text` or `--file`; pass `--model PATH`
to load a model from another directory.
