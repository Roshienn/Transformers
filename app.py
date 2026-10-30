import html

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def get_device() -> torch.device:
    """Prefer Apple's Metal backend, with CPU as a portable fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_model():
    device = get_device()
    st.info(f"Checking device: {device} (MPS preferred, CPU fallback).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    st.success(f"Model ready on {device}.")
    return tokenizer, model, device


def build_messages(mode: str, prompt: str) -> list[dict[str, str]]:
    if mode == "Creative Writing":
        instruction = (
            "Write an engaging, original creative story based on the user's idea. "
            "Use vivid details and a satisfying sense of progression. Return only the story."
        )
    else:
        instruction = (
            "Continue the user's passage naturally from exactly where it ends. "
            "Do not repeat any part of the passage and return only the new continuation."
        )

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]


def generate_text(
    tokenizer,
    model,
    device: torch.device,
    mode: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    messages = build_messages(mode, prompt)
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


st.set_page_config(
    page_title="Transformer Demo",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(73, 92, 190, 0.18), transparent 34rem),
            radial-gradient(circle at 100% 14%, rgba(144, 76, 195, 0.14), transparent 30rem),
            #080b14;
        color: #eef1ff;
    }
    .block-container { max-width: 900px; padding: 2.5rem 1.25rem 3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    p, label, .stMarkdown { font-family: 'DM Sans', sans-serif; }
    .hero { text-align: center; margin-bottom: 1.5rem; }
    .eyebrow {
        color: #9da9ff; font-size: .72rem; font-weight: 700;
        letter-spacing: .15em; text-transform: uppercase; margin-bottom: .55rem;
    }
    .hero h1 { font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.08; margin: 0 0 .55rem; }
    .hero h1 span { color: #9da9ff; }
    .hero p { color: #a7aec4; font-size: 1rem; margin: 0 auto; max-width: 560px; }
    .card {
        background: rgba(18, 23, 39, .78); border: 1px solid #252c48;
        border-radius: 18px; padding: 1.25rem; margin-bottom: 1rem;
        box-shadow: 0 14px 42px rgba(0, 0, 0, .18);
    }
    .section-label {
        color: #c7cdf2; font-family: 'DM Sans', sans-serif;
        font-size: .76rem; font-weight: 700; letter-spacing: .09em;
        text-transform: uppercase; margin: 0 0 .55rem;
    }
    .info-card { color: #a7aec4; font-size: .92rem; line-height: 1.7; }
    .info-card strong { color: #dce1ff; }
    [data-testid="stTextArea"] textarea {
        background: #0e1322; border: 1px solid #2c3557; border-radius: 12px;
        color: #f5f6ff; font-size: 1rem; line-height: 1.55; min-height: 150px;
    }
    [data-testid="stTextArea"] textarea:focus { border-color: #8895ff; box-shadow: 0 0 0 1px #8895ff; }
    [data-testid="stSelectbox"] > div > div, [data-testid="stSlider"] { color: #e8ebff; }
    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stTextArea"]) {
        margin-top: 0;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: .8rem;
        align-items: end;
    }
    div[data-testid="stHorizontalBlock"] > div {
        padding-right: 0 !important;
    }
    .stButton > button {
        width: 100%; border: 0; border-radius: 12px; min-height: 2.8rem;
        background: linear-gradient(110deg, #7181ff, #9b6ade); color: white;
        font-family: 'DM Sans', sans-serif; font-weight: 700; font-size: .96rem;
        box-shadow: 0 10px 24px rgba(113, 129, 255, .22);
    }
    .stButton > button:hover { border: 0; filter: brightness(1.12); color: white; }
    .result-title { color: #9da9ff; font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .6rem; }
    .result-text {
        background: linear-gradient(180deg, rgba(16, 21, 34, 0.95), rgba(12, 17, 29, 1));
        border-left: 3px solid #8290ff; border-radius: 0 12px 12px 0;
        color: #edf0ff; font-family: 'DM Sans', sans-serif; font-size: 1.03rem;
        line-height: 1.75; padding: 1.1rem 1.25rem; white-space: pre-wrap;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>Transformer <span>Demo</span></h1></div>',
    unsafe_allow_html=True,
)

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

st.markdown('<div class="section-label">Your prompt</div>', unsafe_allow_html=True)
prompt = st.text_area(
    "Prompt",
    placeholder="A lighthouse keeper discovers a message written in the newspaper...",
    height=150,
    label_visibility="collapsed",
)

left, right = st.columns(2)
with left:
    mode = st.selectbox("Generation mode", ["Creative Writing", "Continue Text"])
with right:
    temperature = st.slider("Temperature", min_value=0.2, max_value=1.4, value=0.8, step=0.1)

max_new_tokens = st.slider("Maximum new tokens", min_value=32, max_value=512, value=192, step=16)
generate_clicked = st.button("✦  Generate text", type="primary")

if generate_clicked:
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    else:
        try:
            with st.status("Downloading and loading the Transformer...", expanded=True) as status:
                tokenizer, model, device = load_model()
                status.update(label="Model loaded successfully.", state="complete")

            with st.status("Generating text...", expanded=True) as status:
                st.session_state.generated_text = generate_text(
                    tokenizer,
                    model,
                    device,
                    mode,
                    prompt.strip(),
                    temperature,
                    max_new_tokens,
                )
                status.update(label="Text generated successfully.", state="complete")
        except Exception as error:
            st.session_state.generated_text = ""
            st.error(f"Generation failed: {error}")

if st.session_state.generated_text:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Generated result</div><br>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-text">{html.escape(st.session_state.generated_text)}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Clear output"):
        st.session_state.generated_text = ""
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

