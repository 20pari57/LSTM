from pathlib import Path
import html
import pickle

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"
DATASET_PATH = BASE_DIR / "qoute_dataset.csv"


st.set_page_config(
    page_title="LSTM Quote Generator",
    page_icon="Q",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp {
            background: #f6f7fb;
            color: #18212f;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        .app-title {
            border-left: 5px solid #2f80ed;
            padding-left: 1rem;
            margin-bottom: 1.2rem;
        }

        .app-title h1 {
            font-size: 2.2rem;
            line-height: 1.15;
            margin: 0;
            color: #111827;
        }

        .app-title p {
            margin: 0.35rem 0 0;
            color: #526174;
            font-size: 1rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.5rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e4e7ef;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 24px rgba(17, 24, 39, 0.04);
        }

        .metric-label {
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: #111827;
            font-size: 1.55rem;
            font-weight: 700;
        }

        .quote-box {
            background: #ffffff;
            border: 1px solid #d9e0eb;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            min-height: 190px;
            box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
            font-size: 1.08rem;
            line-height: 1.75;
            color: #172033;
            overflow-wrap: anywhere;
        }

        .status-ok {
            background: #e8f7ef;
            border: 1px solid #b9e7cc;
            color: #0f6b3f;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            font-weight: 600;
        }

        .status-warn {
            background: #fff7e6;
            border: 1px solid #ffd999;
            color: #8a5700;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            font-weight: 600;
        }

        div.stButton > button {
            border-radius: 8px;
            border: 1px solid #2f80ed;
            background: #2f80ed;
            color: white;
            font-weight: 700;
            min-height: 2.7rem;
        }

        div.stButton > button:hover {
            border-color: #1f6ed4;
            background: #1f6ed4;
            color: white;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e4e7ef;
            border-radius: 8px;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            .metric-strip {
                grid-template-columns: 1fr;
            }

            .app-title h1 {
                font-size: 1.65rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_quote(text: object) -> str:
    """Normalize common quote mojibake and spacing in the dataset/model output."""
    value = "" if pd.isna(text) else str(text)
    replacements = {
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u201d": "-",
        "\ufffd": '"',
    }

    for bad, good in replacements.items():
        value = value.replace(bad, good)

    return " ".join(value.strip().split())


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        return pd.DataFrame(columns=["quote", "Author"])

    data = pd.read_csv(DATASET_PATH)
    data.columns = [column.strip() for column in data.columns]

    if "quote" in data.columns:
        data["quote"] = data["quote"].apply(clean_quote)

    if "Author" in data.columns:
        data["Author"] = data["Author"].fillna("Unknown").astype(str).str.strip()

    return data


def load_keras_model():
    try:
        from keras.models import load_model

        return load_model
    except Exception:
        try:
            from tensorflow.keras.models import load_model

            return load_model
        except Exception as exc:
            raise RuntimeError(
                "Keras/TensorFlow is not installed. Run: pip install tensorflow keras"
            ) from exc


@st.cache_resource(show_spinner=False)
def load_model_assets():
    required_files = [MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH]
    missing = [path.name for path in required_files if not path.exists()]

    if missing:
        return {
            "ready": False,
            "error": f"Missing project file(s): {', '.join(missing)}",
        }

    try:
        load_model = load_keras_model()

        with TOKENIZER_PATH.open("rb") as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        with MAX_LEN_PATH.open("rb") as max_len_file:
            max_len = int(pickle.load(max_len_file))

        model = load_model(MODEL_PATH, compile=False)

        return {
            "ready": True,
            "model": model,
            "tokenizer": tokenizer,
            "max_len": max_len,
        }
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
        }


def model_input_length(model, max_len: int) -> int:
    shape = getattr(model, "input_shape", None)

    if isinstance(shape, list) and shape:
        shape = shape[0]

    if shape and len(shape) > 1 and shape[1]:
        return int(shape[1])

    return max(1, int(max_len) - 1)


def pad_tokens(tokens: list[int], target_length: int) -> np.ndarray:
    padded = np.zeros((1, target_length), dtype=np.int32)
    clipped = tokens[-target_length:]

    if clipped:
        padded[0, -len(clipped) :] = clipped

    return padded


def sample_next_word(predictions: np.ndarray, temperature: float, top_k: int) -> int:
    probabilities = np.asarray(predictions, dtype=np.float64).reshape(-1)
    probabilities = np.maximum(probabilities, 1e-12)

    if top_k > 0 and top_k < len(probabilities):
        candidate_indexes = probabilities.argsort()[-top_k:]
        candidate_probabilities = probabilities[candidate_indexes]
    else:
        candidate_indexes = np.arange(len(probabilities))
        candidate_probabilities = probabilities

    logits = np.log(candidate_probabilities) / max(temperature, 1e-6)
    scaled = np.exp(logits - np.max(logits))
    scaled = scaled / scaled.sum()

    return int(np.random.choice(candidate_indexes, p=scaled))


def generate_quote(
    seed_text: str,
    next_words: int,
    temperature: float,
    top_k: int,
    assets: dict,
) -> str:
    model = assets["model"]
    tokenizer = assets["tokenizer"]
    max_len = assets["max_len"]
    input_length = model_input_length(model, max_len)
    generated = clean_quote(seed_text)
    index_word = getattr(tokenizer, "index_word", None) or {
        index: word for word, index in tokenizer.word_index.items()
    }

    for _ in range(next_words):
        tokens = tokenizer.texts_to_sequences([generated])[0]

        if not tokens:
            break

        padded_tokens = pad_tokens(tokens, input_length)
        predictions = model.predict(padded_tokens, verbose=0)
        next_index = sample_next_word(predictions[0], temperature, top_k)
        next_word = index_word.get(next_index)

        if not next_word:
            break

        generated = f"{generated} {next_word}"

    return clean_quote(generated)


dataset = load_dataset()
assets = load_model_assets()

quote_count = len(dataset)
author_count = dataset["Author"].nunique() if "Author" in dataset.columns else 0
vocab_count = (
    len(getattr(assets.get("tokenizer"), "word_index", {})) if assets.get("ready") else 0
)

st.markdown(
    """
    <div class="app-title">
        <h1>LSTM Quote Generator</h1>
        <p>Create quote-style text from your trained LSTM model and explore the training dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-strip">
        <div class="metric-card">
            <div class="metric-label">Quotes</div>
            <div class="metric-value">{quote_count:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Authors</div>
            <div class="metric-value">{author_count:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Vocabulary</div>
            <div class="metric-value">{vocab_count:,}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project")

    if assets.get("ready"):
        st.markdown('<div class="status-ok">Model loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-warn">Model not loaded</div>',
            unsafe_allow_html=True,
        )
        st.caption(assets.get("error", "Unknown loading error"))

    st.divider()
    st.write("Files")
    st.checkbox("lstm_model.h5", value=MODEL_PATH.exists(), disabled=True)
    st.checkbox("tokenizer.pkl", value=TOKENIZER_PATH.exists(), disabled=True)
    st.checkbox("max_len.pkl", value=MAX_LEN_PATH.exists(), disabled=True)
    st.checkbox("qoute_dataset.csv", value=DATASET_PATH.exists(), disabled=True)

    st.divider()
    st.write("Run")
    st.code("streamlit run app.py", language="bash")


generator_tab, dataset_tab = st.tabs(["Generate", "Dataset"])

with generator_tab:
    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        seed_text = st.text_area(
            "Start text",
            value="life is",
            height=120,
            placeholder="Type a few words...",
        )

        next_words = st.slider("Words to generate", 5, 80, 25, 5)
        temperature = st.slider("Creativity", 0.2, 1.5, 0.8, 0.1)
        top_k = st.slider("Top-k choices", 1, 25, 5, 1)

        generate_clicked = st.button(
            "Generate quote",
            type="primary",
            use_container_width=True,
            disabled=not assets.get("ready"),
        )

        if not assets.get("ready"):
            st.warning("Install TensorFlow/Keras to enable generation.")
            st.code("pip install tensorflow keras", language="bash")

    with right:
        st.subheader("Output")

        if "generated_quote" not in st.session_state:
            st.session_state.generated_quote = (
                "Your generated quote will appear here after the model is ready."
            )

        if generate_clicked:
            if seed_text.strip():
                with st.spinner("Generating..."):
                    st.session_state.generated_quote = generate_quote(
                        seed_text=seed_text,
                        next_words=next_words,
                        temperature=temperature,
                        top_k=top_k,
                        assets=assets,
                    )
            else:
                st.session_state.generated_quote = "Please enter start text first."

        st.markdown(
            f'<div class="quote-box">{html.escape(st.session_state.generated_quote)}</div>',
            unsafe_allow_html=True,
        )

        if assets.get("ready"):
            st.caption(
                "Lower creativity is more predictable; higher creativity is more varied."
            )


with dataset_tab:
    st.subheader("Training Quotes")

    if dataset.empty:
        st.info("Dataset file was not found or is empty.")
    else:
        author_options = ["All authors"] + sorted(dataset["Author"].dropna().unique())

        filter_col, search_col = st.columns([0.8, 1.2], gap="large")

        with filter_col:
            selected_author = st.selectbox("Author", author_options)

        with search_col:
            search_text = st.text_input("Search", placeholder="Find words in quotes")

        filtered = dataset.copy()

        if selected_author != "All authors":
            filtered = filtered[filtered["Author"] == selected_author]

        if search_text.strip():
            filtered = filtered[
                filtered["quote"].str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        st.caption(f"Showing {len(filtered):,} of {len(dataset):,} quotes")
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=420)
