"""Streamlit user interface for SentiSense."""

from collections import Counter

import pandas as pd
import streamlit as st

from sentiment import CLASS_ORDER, SentimentAnalyzer


@st.cache_resource
def load_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


st.set_page_config(page_title="SentiSense", page_icon="💬")
st.title("SentiSense")
st.write("Enter a message and the model will estimate its sentiment.")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

text = st.text_area("Text to analyze", max_chars=10_000, height=160)
if st.button("Analyze text", type="primary"):
    if not text.strip():
        st.warning("Enter some text to analyze.")
    else:
        try:
            result = load_analyzer().predict(text)
            st.session_state.analysis_history.append(
                {
                    "text": text.strip(),
                    "sentiment": result["sentiment"],
                    "confidence": result["score"],
                }
            )
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))

history = st.session_state.analysis_history
if history:
    latest = history[-1]
    st.subheader(f"Latest result: {latest['sentiment'].title()}")
    st.metric("Confidence", f"{latest['confidence']:.1%}")

    counts = Counter(entry["sentiment"] for entry in history)
    total = len(history)
    summary = pd.DataFrame(
        {
            "Sentiment": [label.title() for label in CLASS_ORDER],
            "Entries": [counts[label] for label in CLASS_ORDER],
            "Overall rate (%)": [counts[label] / total * 100 for label in CLASS_ORDER],
        }
    )

    st.subheader(f"Overall sentiment across {total} entr{'y' if total == 1 else 'ies'}")
    chart_data = summary.set_index("Sentiment")[["Overall rate (%)"]]
    st.bar_chart(chart_data)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    with st.expander("Analyzed entries"):
        entries = pd.DataFrame(
            [
                {
                    "#": index,
                    "Text": entry["text"],
                    "Sentiment": entry["sentiment"].title(),
                    "Confidence": f"{entry['confidence']:.1%}",
                }
                for index, entry in enumerate(history, start=1)
            ]
        )
        st.dataframe(entries, hide_index=True, use_container_width=True)
else:
    st.info("Analyze a text to start building your session graph.")
