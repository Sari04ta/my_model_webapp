# streamlit_app.py
"""
Streamlit app for your fine-tuned Flan-T5 summarizer stored in ./final-tuned-summarizer

Behavior:
- If ./final-tuned-summarizer exists in the repo, the app loads it locally.
- Otherwise, you can enter a Hugging Face repo id to load from the Hub.
- Simple, clean UI with generation controls and batch upload support.
"""

from pathlib import Path
import streamlit as st

# Lazy imports so page loads and shows friendly messages while heavy libs download
@st.cache_resource
def load_model_and_tokenizer(model_path_or_name: str, device: str = "cpu"):
    """
    model_path_or_name: local folder (./final-tuned-summarizer) or HF repo id (e.g. username/repo)
    device: "cpu" or "cuda"
    Returns (tokenizer, model)
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_name)
    if device == "cuda" and torch.cuda.is_available():
        model.to(torch.device("cuda"))
    return tokenizer, model

def generate_one(tokenizer, model, prompt: str, max_length=128, num_beams=4, temperature=1.0, do_sample=False):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    # move to model device
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    gen_kwargs = dict(
        max_length=max_length,
        num_beams=num_beams if not do_sample else 1,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    # remove None
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return decoded[0]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PAP — Summarizer", layout="wide")
st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg, #0f172a 0%, #071032 100%); color: #e6eef8; }
    .card { background: rgba(255,255,255,0.03); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.04); }
    .title { font-size:28px; font-weight:700; }
    .muted { color:#9aa7c7; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">PAP — Fine-tuned Summarizer (Flan-T5)</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Load your local fine-tuned model or a Hugging Face repo. Enter text or upload a file.</div>', unsafe_allow_html=True)
st.write("")

# Sidebar controls
with st.sidebar:
    st.header("Model settings")
    local_model_folder = st.text_input("Local model folder (repo)", value="final-tuned-summarizer")
    hf_fallback = st.text_input("Or HF repo id (optional)", value="")
    use_gpu = st.checkbox("Use GPU if available", value=False)
    device = "cuda" if use_gpu else "cpu"
    st.markdown("---")
    st.header("Generation defaults")
    default_max_length = st.slider("Max length", min_value=16, max_value=512, value=128, step=8)
    default_num_beams = st.slider("num_beams", min_value=1, max_value=8, value=4)
    default_temperature = st.slider("temperature (sampling only)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    sampling_default = st.checkbox("Enable sampling by default", value=False)
    st.markdown("---")
    st.markdown("Deployment notes:")
    st.write("- If your model folder is large, consider using Git LFS or uploading to Hugging Face Hub.")
    st.write("- Streamlit Cloud may not provide GPU by default.")

# Determine model source
local_path = Path(local_model_folder)
model_source = None
if local_path.exists() and local_path.is_dir():
    model_source = str(local_path)
elif hf_fallback.strip():
    model_source = hf_fallback.strip()
else:
    st.warning("No local model folder found in the repo. Enter a HuggingFace repo id in the sidebar to load from the Hub.")
    st.info("If you plan to add your local model folder to the repo, name it exactly: final-tuned-summarizer and push it to GitHub (see README instructions).")

# Load model (if available)
tokenizer, model = None, None
if model_source:
    with st.spinner(f"Loading model from {model_source} ... (this may take a bit)"):
        try:
            tokenizer, model = load_model_and_tokenizer(model_source, device=device)
            st.success(f"Model loaded from: {model_source}")
            # Show device
            try:
                import torch
                st.write("Device:", next(model.parameters()).device)
            except Exception:
                pass
        except Exception as e:
            st.error("Failed to load model. See error below:")
            st.text(str(e))
            st.stop()

# Main layout
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Enter text or prompt")
    text_input = st.text_area("Input text (or prompt)", height=240, placeholder="Paste the text to summarize here...")
    st.markdown("Or upload a .txt file with lines of prompts (batch).")
    uploaded = st.file_uploader("Upload a .txt file (one input per line)", type=["txt"])
    st.write("")

    st.subheader("Generation controls (overrides sidebar values)")
    max_length = st.number_input("Max length", min_value=16, max_value=1024, value=default_max_length)
    num_beams = st.number_input("num_beams", min_value=1, max_value=16, value=default_num_beams)
    temperature = st.number_input("temperature", min_value=0.1, max_value=2.0, value=default_temperature, step=0.1)
    do_sample = st.checkbox("Use sampling (stochastic)", value=sampling_default)
    run = st.button("Summarize")

    st.markdown("---")
    st.markdown("Output")
    output_container = st.empty()

with col2:
    st.markdown("## Examples & Info")
    st.markdown("**Example prompts**")
    st.write("- Summarize: Provide a concise summary in 2 sentences.")
    st.write("- Clean: Remove persuasive/unsafe intent and make neutral.")
    st.markdown("---")
    st.markdown("Model source:")
    st.code(model_source or "None")
    st.markdown("Local folder contents (if available):")
    if model_source and Path(model_source).exists():
        for p in sorted(Path(model_source).iterdir()):
            st.write(p.name)
    st.markdown("---")
    st.markdown("Notes")
    st.write("- If the model is large (>100MB) use Git LFS or push to HuggingFace Hub for reliable deployment.")

# Run generation
if (run or uploaded is not None) and tokenizer is not None and model is not None:
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    else:
        lines = [text_input.strip()] if text_input.strip() else []

    if not lines:
        st.warning("No input text provided.")
    else:
        results = []
        with st.spinner("Generating..."):
            for i, ln in enumerate(lines):
                # prefix: you may want to prepend a task instruction like 'summarize: '
                prompt = f"summarize: {ln}"
                try:
                    out = generate_one(tokenizer, model, prompt, max_length=int(max_length),
                                       num_beams=int(num_beams), temperature=float(temperature), do_sample=do_sample)
                    results.append((ln, out))
                except Exception as e:
                    results.append((ln, f"Generation failed: {e}"))

        # show results
        for i, (inp, out) in enumerate(results):
            st.markdown(f"### Result {i+1}")
            st.markdown("**Input:**")
            st.write(inp if len(inp) < 2000 else inp[:2000] + "...")
            st.markdown("**Output:**")
            st.write(out)

st.markdown("---")
st.markdown("## Deployment tips")
st.write("1) If your local model folder is large (>100MB files), use Git LFS or upload to Hugging Face Hub and provide the repo id in the sidebar.")
st.write("2) Streamlit Cloud will install packages from requirements.txt. First deploy may take 1-3 minutes to install torch/transformers.")
