import io
import re
import time
import streamlit as st

# --- Optional libraries for PDF text extraction ---
try:
    import pdfplumber  # better text extraction for digital PDFs
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    from pdf2image import convert_from_bytes  # for OCR fallback
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False

# --- Ollama (local LLM) ---
# pip install ollama
try:
    import ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

st.set_page_config(page_title="Chat with Docs (Ollama)", layout="wide")
st.title("🧠 Chat with Docs")

# =========================
# Sidebar
# =========================
st.sidebar.caption("💡 Jalankan: `streamlit run chat-by-streamlit/chat-docs-app.py`")
st.sidebar.caption("🧱 Pastikan Ollama aktif & model sudah di-pull, contoh: `ollama pull gemma3:1b`")
st.sidebar.markdown("---")
use_llm = st.sidebar.checkbox("Gunakan Ollama", value=True)
default_model = "gemma3:1b"  # ganti sesuai model yang sudah di-pull di Ollama
ollama_host = st.sidebar.text_input("Ollama Host", value="http://localhost:11434")
model_name = st.sidebar.text_input("Model Ollama", value=default_model)
uploaded_file = st.file_uploader("📤 Unggah File PDF", type=["pdf"])

# =========================
# Helper Functions
# =========================
def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def clean_answer(text: str) -> str:
    # hapus blok <think> ... </think> untuk model reasoning
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def pdf_text_first_then_ocr(file_bytes: bytes, ocr_if_short: bool = True, min_chars: int = 3000) -> str:
    text = ""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = []
                for p in pdf.pages:
                    pages.append(p.extract_text() or "")
                text = "\n".join(pages)
        except Exception as e:
            st.warning(f"Gagal ekstraksi teks via pdfplumber: {e}")

    need_ocr = (len(text) < min_chars) if ocr_if_short else False
    if need_ocr and HAS_OCR:
        st.info("🖼️ Teks minim — fallback ke OCR (ini bisa lebih lama)...")
        try:
            images = convert_from_bytes(file_bytes, dpi=300)
            ocr_chunks = []
            for idx, img in enumerate(images, 1):
                ocr_chunks.append(pytesseract.image_to_string(img, lang="ind+eng"))
                if idx % 3 == 0:
                    time.sleep(0.1)
            text = "\n".join(ocr_chunks)
        except Exception as e:
            st.error(f"Gagal OCR: {e}")
    elif need_ocr and not HAS_OCR:
        st.warning("OCR dibutuhkan tapi `pdf2image`/`pytesseract` belum tersedia. Install: `pip install pdf2image pytesseract` dan pasang Tesseract di OS.")

    return text

# =========================
# Main
# =========================
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.info("🔍 Ekstraksi teks dari PDF... (text-first, OCR fallback)")
    full_text = pdf_text_first_then_ocr(file_bytes)
    full_text = full_text or ""

    st.subheader("📄 Teks Mentah (Hasil Ekstraksi)")
    st.text_area("Hasil Ekstraksi", full_text, height=300)

    # simpan teks dokumen di session_state supaya bisa dipakai terus
    st.session_state["doc_text"] = full_text

else:
    st.info("📂 Unggah PDF untuk mulai ekstraksi dan chat.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Chat dengan Dokumen")

# input chat user
user_input = st.chat_input("Tanyakan sesuatu dari dokumen...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if use_llm and HAS_OLLAMA:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "Anda adalah asisten yang menjawab berdasarkan isi dokumen."},
            {"role": "user", "content": f"Berikut adalah dokumen:\n{st.session_state['doc_text'][:200000]}"},
        ]
        messages.extend(st.session_state.chat_history)

        try:
            resp = client.chat(model=model_name, messages=messages, stream=False)
            answer = resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            answer = f"(Error chat dengan Ollama: {e})"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# tampilkan percakapan
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(clean_answer(msg["content"]))

