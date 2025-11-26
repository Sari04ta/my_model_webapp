# streamlit_app.py
"""
Streamlit UI for the PAP model using HuggingFace model 'google/flan-t5-small'.

Features:
- Loads the model/tokenizer from transformers by model name.
- Text input box (prompt) and controls (max_length, temperature, num_beams).
- Shows generation result, token count, and raw model output.
- Simple caching so model loads once per process.
- Optional: accept uploaded text files for batch inference.

To run:
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

from pathlib import Path
import streamlit as st

# Transformers & torch imports are done lazily to show friendly error messages
@st.cache_resource
def load_model_tokenizer(model_name: str, device: int = -1):
    """
    Load the seq2seq model and tokenizer. device = -1 uses CPU; 0 uses first GPU.
    Returns (tokenizer, model, pipeline_fn)
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Move to device if GPU available
    if device >= 0 and torch.cuda.is_available():
        model.to(torch.device(f"cuda:{device}"))
    return tokenizer, model

def generate_text(tokenizer, model, prompt: str, max_length: int = 128,
                  num_beams: int = 4, temperature: float = 1.0, do_sample: bool = False,
                  return_full: bool = False):
    """
    Generate text from the model. Returns generated string (and optionally generation details).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to model device
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    # generation kwargs
    gen_kwargs = dict(
        max_length=max_length,
        num_beams=num_beams if not do_sample else 1,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    # Remove None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # Generate
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    if return_full:
        return decoded, out
    return decoded[0]

# ---------------------------
# Page layout & UI
# ---------------------------
st.set_page_config(page_title="PAP — Flan-T5 demo", layout="wide")
st.markdown(
    """
    <style>
    .big-title { font-size:32px; font-weight:700; }
    .muted { color: #bdbdbd; }
    .card { background: rgba(255,255,255,0.03); padding:16px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">PAP — Flan-T5 (google/flan-t5-small) Demo</div>', unsafe_allow_html=True)
st.markdown("Interactive demo — enter a prompt and get model output. Deployable on Streamlit Cloud.")

# Sidebar controls
with st.sidebar:
    st.header("Model & Settings")
    model_name = st.text_input("HuggingFace model name", value="google/flan-t5-small")
    use_gpu = st.checkbox("Use GPU if available", value=False)
    device_idx = 0 if use_gpu else -1
    st.markdown("---")
    st.markdown("Generation defaults")
    default_max_length = st.slider("Max length", min_value=16, max_value=512, value=128, step=8)
    default_num_beams = st.slider("num_beams (deterministic search)", min_value=1, max_value=8, value=4)
    default_temperature = st.slider("temperature (for sampling)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    do_sample = st.checkbox("Use sampling (stochastic)", value=False)
    st.markdown("---")
    st.markdown("Advanced")
    show_raw = st.checkbox("Show raw tokens / outputs", value=False)
    st.markdown("Note: first load may take a few seconds to download model files.")

# Main columns
col_left, col_right = st.columns([2,1])
with col_left:
    st.subheader("Prompt")
    prompt = st.text_area("Enter your prompt or input text", height=220,
                          value="Translate English to French: The climate is changing rapidly. We must act now.")
    max_length = st.number_input("Max tokens to generate", min_value=16, max_value=1024, value=default_max_length)
    num_beams = st.number_input("num_beams", min_value=1, max_value=16, value=default_num_beams)
    temperature = st.number_input("temperature", min_value=0.1, max_value=2.0, value=default_temperature, step=0.1)
    do_sample = st.checkbox("Use sampling", value=do_sample)
    run_btn = st.button("Generate")

    st.markdown("---")
    st.markdown("Batch input (optional)")
    uploaded = st.file_uploader("Upload a text file (.txt) with multiple prompts (one per line)", type=["txt"])
    if uploaded is not None:
        content = uploaded.getvalue().decode("utf-8")
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        st.success(f"Loaded {len(lines)} prompts from file.")
        # show a small sample
        for i, line in enumerate(lines[:5]):
            st.write(f"{i+1}. {line[:120]}{'...' if len(line) > 120 else ''}")

with col_right:
    st.subheader("Model status")
    st.write("Model name:")
    st.code(model_name)
    try:
        st.write("Loading model...")
        tokenizer, model = load_model_tokenizer(model_name, device=device_idx)
        st.success("Model loaded.")
        # show device
        try:
            import torch
            device = next(model.parameters()).device
            st.write("Device:", device)
        except Exception:
            pass
    except Exception as e:
        st.error("Failed to load model. See error below.")
        st.text(str(e))
        st.stop()

# Run generation
if run_btn or (uploaded is not None):
    with st.spinner("Generating..."):
        try:
            if uploaded is not None:
                # batch generate
                results = []
                for ln in lines:
                    out = generate_text(tokenizer, model, ln, max_length=max_length,
                                        num_beams=int(num_beams), temperature=float(temperature),
                                        do_sample=do_sample)
                    results.append((ln, out))
                st.markdown("### Batch results")
                for i, (inp, out) in enumerate(results):
                    st.markdown(f"**Prompt {i+1}:** {inp}")
                    st.markdown(f"**Output:** {out}")
                    st.markdown("---")
            else:
                out = generate_text(tokenizer, model, prompt, max_length=max_length,
                                    num_beams=int(num_beams), temperature=float(temperature),
                                    do_sample=do_sample)
                st.markdown("### Output")
                st.write(out)
                if show_raw:
                    st.markdown("### Raw details")
                    decoded, tok = None, None
                    try:
                        decoded, tok = generate_text(tokenizer, model, prompt, max_length=max_length,
                                                     num_beams=int(num_beams), temperature=float(temperature),
                                                     do_sample=do_sample, return_full=True)
                    except Exception:
                        pass
                    if decoded is not None:
                        st.write("Decoded:", decoded)
                    if tok is not None:
                        st.write("Tensor shape:", getattr(tok, "shape", None))
        except Exception as e:
            st.error("Generation failed. See traceback below.")
            import traceback as tb
            st.text(tb.format_exc())
