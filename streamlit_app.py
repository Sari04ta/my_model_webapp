# streamlit_app.py
"""
Beautiful Streamlit app that loads your model from two notebooks:
- Final model notebook: /mnt/data/pap (1).ipynb
- Prototype notebook (helpers/UI examples): /mnt/data/prototype (1).ipynb

Behavior:
- Import code cells from both notebooks into separate temporary modules.
- Prefer predictor from the final notebook; if missing, fall back to prototype.
- Auto-detect `feature_names` or `X_example` from either notebook to build a form.
- Accept raw JSON/text input or file upload for model.pkl.
- Show prediction + debug information.

How to run:
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

from pathlib import Path
import nbformat
import importlib.util
import sys
import tempfile
import traceback
import inspect
import types
import streamlit as st
import ast
import io
import pickle

# === Paths to user notebooks (these are the files you uploaded) ===
FINAL_NOTEBOOK = Path("pap.ipynb")
PROTO_NOTEBOOK = Path("prototype.ipynb")


# === Page config & CSS ===
st.set_page_config(page_title="PAP Model — Demo", layout="wide")
st.markdown(
    """
    <style>
    /* Glass card */
    .glass {
      background: rgba(255, 255, 255, 0.06);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
    }
    .muted { color: #bdbdbd; font-size: 0.95rem; }
    .big { font-size: 1.25rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([3,1])
with col1:
    st.title("PAP Model — Interactive Demo")
    st.markdown("**Upload-ready Streamlit app** which combines your prototype and final model notebooks.")
with col2:
    st.image("https://static.streamlit.io/examples/dice.jpg", width=100)  # decorative

st.write("")  # spacing

# Sidebar
with st.sidebar:
    st.header("About this demo")
    st.write("Model loaded from your notebooks (final prioritized).")
    st.markdown("**Actions**")
    st.markdown("- Try example inputs\n- Upload `model.pkl` to override\n- Use Raw JSON input")
    st.markdown("---")
    st.markdown("**Files used**:")
    st.text(FINAL_NOTEBOOK)
    st.text(PROTO_NOTEBOOK)
    st.markdown("---")
    st.markdown("Need a FastAPI + React production API? Ask me to generate it.")

# Utility: import notebook cells as a module
def import_notebook_module(nb_path: Path, module_name: str):
    """
    Read code cells from nb_path, write to a temp .py and import as module_name.
    Returns module or raises an error.
    """
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    nb = nbformat.read(str(nb_path), as_version=4)
    code_cells = [cell['source'] for cell in nb.cells if cell.cell_type == 'code']
    combined = "\n\n# ---- cell split ----\n\n".join(code_cells)
    tmp_dir = tempfile.mkdtemp(prefix=f"uploaded_nb_{module_name}_")
    module_path = Path(tmp_dir) / f"{module_name}.py"
    module_path.write_text(combined, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Import may still have created objects partially; return module with note
        module.__import_error__ = traceback.format_exc()
    return module

# Import both notebooks (final first, prototype second)
final_module = None
proto_module = None
final_import_error = None
proto_import_error = None

try:
    final_module = import_notebook_module(FINAL_NOTEBOOK, "uploaded_nb_final")
    final_import_error = getattr(final_module, "__import_error__", None)
except Exception as e:
    final_import_error = traceback.format_exc()
    final_module = types.SimpleNamespace()  # empty fallback

try:
    proto_module = import_notebook_module(PROTO_NOTEBOOK, "uploaded_nb_proto")
    proto_import_error = getattr(proto_module, "__import_error__", None)
except Exception as e:
    proto_import_error = traceback.format_exc()
    proto_module = types.SimpleNamespace()

# Helper: find predictor in a module
def find_predictor_in_module(module):
    # find predict() function
    if hasattr(module, "predict") and callable(getattr(module, "predict")):
        return getattr(module, "predict"), "function"
    # find model object
    if hasattr(module, "model"):
        model_obj = getattr(module, "model")
        if hasattr(model_obj, "predict") and callable(getattr(model_obj, "predict")):
            return (lambda x: model_obj.predict(x)), "model"
    # try any object with predict
    for name, obj in vars(module).items():
        if not name.startswith("_") and hasattr(obj, "predict") and callable(getattr(obj, "predict")):
            return (lambda x, o=obj: o.predict(x)), f"object `{name}`"
    return None, None

# Try final, then prototype
predictor, predictor_type = find_predictor_in_module(final_module)
source_used = "final"
if predictor is None:
    predictor, predictor_type = find_predictor_in_module(proto_module)
    source_used = "prototype" if predictor else None

# UI: show import status
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("### Model import status")
    if final_import_error:
        st.warning("Final notebook import produced warnings/errors. Predictor may still be available.")
        st.text(final_import_error[:1000])
    if proto_import_error:
        st.info("Prototype notebook import produced warnings/errors (expected sometimes).")
        st.text(proto_import_error[:800])
    if predictor:
        st.success(f"Predictor found (source: `{source_used}`, type: {predictor_type})")
    else:
        st.error("No predictor found in final or prototype notebooks.")
        st.markdown("Make sure either notebook defines `predict(input)` or a `model` object with `.predict()`.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # space

# Gather metadata (feature_names, X_example) from final then prototype
feature_names = getattr(final_module, "feature_names", None) or getattr(proto_module, "feature_names", None)
X_example = getattr(final_module, "X_example", None) or getattr(proto_module, "X_example", None)
examples = getattr(proto_module, "examples", None) or getattr(final_module, "examples", None)

# Main two-column layout
left, right = st.columns([2,1])

with left:
    st.markdown("## Input")
    mode = st.radio("Input mode", ["Auto form", "Raw JSON / Text", "Upload model.pkl"], horizontal=True)

    user_input = None
    uploaded_model_override = None

    if mode == "Upload model.pkl":
        st.info("Upload a pickle file containing a model with `.predict()` to override the notebook model.")
        up = st.file_uploader("Upload model.pkl", type=["pkl","pickle"])
        if up is not None:
            try:
                uploaded_model_override = pickle.load(up)
                if hasattr(uploaded_model_override, "predict") and callable(uploaded_model_override.predict):
                    predictor = (lambda x, o=uploaded_model_override: o.predict(x))
                    predictor_type = "uploaded model.pkl"
                    st.success("Uploaded model loaded and will be used for predictions.")
                else:
                    st.error("Uploaded object does not have a callable `predict` method.")
            except Exception as e:
                st.error("Failed to load uploaded pickle.")
                st.text(traceback.format_exc()[:800])

    if mode == "Auto form":
        # Build form from feature_names or X_example, else fallback
        if feature_names and isinstance(feature_names, (list, tuple)):
            st.write("Detected feature names — fill the fields below.")
            inputs = {}
            with st.form("auto_form"):
                for fn in feature_names:
                    # prefer numeric
                    try:
                        val = st.number_input(str(fn), value=0.0, format="%.6f")
                    except Exception:
                        val = st.text_input(str(fn), value="")
                    inputs[fn] = val
                submitted = st.form_submit_button("Run prediction")
            if submitted:
                # prepare input_data as a list or dict depending on what predictor expects
                # We'll try list first then dict
                input_as_list = [inputs[n] for n in feature_names]
                input_as_dict = {n: inputs[n] for n in feature_names}
                try:
                    # attempt list
                    user_input = input_as_list
                    pred = predictor(user_input)
                    st.success("Prediction:")
                    st.write(pred)
                except Exception:
                    try:
                        user_input = input_as_dict
                        pred = predictor(user_input)
                        st.success("Prediction:")
                        st.write(pred)
                    except Exception:
                        st.error("Both list and dict attempts failed. See traceback below.")
                        st.text(traceback.format_exc()[:1200])

        elif X_example is not None:
            st.write("Detected `X_example` in notebook — using it to build fields.")
            inputs = {}
            if isinstance(X_example, dict):
                with st.form("dict_example_form"):
                    for k, v in X_example.items():
                        if isinstance(v, (int, float)):
                            val = st.number_input(str(k), value=float(v))
                        else:
                            val = st.text_input(str(k), value=str(v))
                        inputs[k] = val
                    submitted = st.form_submit_button("Run prediction")
                if submitted:
                    try:
                        user_input = {k: inputs[k] for k in inputs}
                        pred = predictor(user_input)
                        st.success("Prediction:")
                        st.write(pred)
                    except Exception:
                        st.error("Prediction failed with dict input. See traceback.")
                        st.text(traceback.format_exc()[:800])
            elif hasattr(X_example, "__len__"):
                with st.form("vec_example_form"):
                    for i, v in enumerate(X_example):
                        try:
                            val = st.number_input(f"f{i}", value=float(v))
                        except Exception:
                            val = st.text_input(f"f{i}", value=str(v))
                        inputs[f"f{i}"] = val
                    submitted = st.form_submit_button("Run prediction")
                if submitted:
                    try:
                        user_input = [inputs[f"f{i}"] for i in range(len(X_example))]
                        pred = predictor(user_input)
                        st.success("Prediction:")
                        st.write(pred)
                    except Exception:
                        st.error("Prediction failed with vector input. See traceback.")
                        st.text(traceback.format_exc()[:800])
            else:
                st.write("X_example format not recognized — fallback to Raw JSON mode.")
        else:
            st.write("No `feature_names` or `X_example` detected. Use Raw JSON / Text input or upload a model.pkl.")
    elif mode == "Raw JSON / Text":
        raw = st.text_area("Enter JSON, Python literal, or plain text input", height=220,
                           placeholder='e.g. [1.0, 2.0, 3.0]  or  {"age": 20, "bmi": 23.4}')
        if st.button("Run prediction (raw)"):
            try:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    import json
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = raw
                user_input = parsed
                pred = predictor(user_input)
                st.success("Prediction:")
                st.write(pred)
            except Exception:
                st.error("Prediction failed. See traceback below.")
                st.text(traceback.format_exc()[:1200])

with right:
    st.markdown("## Controls & Examples")
    if examples:
        st.markdown("### Example inputs (from prototype)")
        if isinstance(examples, (list, tuple)):
            for i, ex in enumerate(examples[:6]):
                if st.button(f"Use example #{i+1}"):
                    try:
                        pred = predictor(ex)
                        st.success("Prediction for example:")
                        st.write(pred)
                    except Exception:
                        st.error("Prediction failed for this example.")
                        st.text(traceback.format_exc()[:800])
        else:
            st.write("`examples` exists but is not a list/tuple — inspect in notebooks.")
    else:
        st.info("No example inputs provided in notebooks. You can add `examples = [...]` variable in prototype notebook.")

    st.markdown("---")
    st.markdown("### Debug / Info")
    st.write(f"Predictor type: {predictor_type}")
    if hasattr(final_module, "__import_error__") and final_module.__import_error__:
        st.markdown("**Final notebook import trace (first 500 chars):**")
        st.text(final_module.__import_error__[:500])
    if hasattr(proto_module, "__import_error__") and proto_module.__import_error__:
        st.markdown("**Prototype notebook import trace (first 500 chars):**")
        st.text(proto_module.__import_error__[:500])

st.markdown("---")
st.markdown("## Notes & Next steps")
st.markdown("""
- If import fails because of heavy dependencies, consider exporting a lightweight `model.pkl` and upload it in the `Upload model.pkl` tab.
- To turn this into a production API, I can create a FastAPI backend and a React frontend, plus Dockerfiles.
""")
