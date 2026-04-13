import streamlit as st
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import io

# Konfiguracja cache (opcjonalna jak jest mało miejsca na dysku C:
# MY_CACHE_DIR = cache_loc
# os.environ['HF_HOME'] = MY_CACHE_DIR


st.set_page_config(page_title="Generator mowy", page_icon="🎙️", layout="wide")

# Mapowanie języków i dostępnych lektorów z dokumentacji
SPEAKERS = {
    "Polski": ["Alex", "Natalie"],
    "English": ["Daniel", "Christine", "Richard", "Nicole", "Gary", "Elizabeth"],
    "German": ["Nicole", "Christopher", "Megan", "Michelle"],
    "French": ["Daniel", "Michelle", "Christine", "Megan"],
    "Italian": ["Julia", "Richard", "Megan"],
    "Spanish": ["Steven", "Olivia", "Megan"],
    "Dutch": ["Mark", "Jessica", "Michelle"],
    "Portuguese": ["Sophia", "Nicholas"]
}


@st.cache_resource
def load_parler():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    repo_id = "parler-tts/parler-tts-mini-multilingual-v1.1"

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        repo_id#, cache_dir=MY_CACHE_DIR
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(repo_id#, cache_dir=MY_CACHE_DIR
                                              )
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path#, cache_dir=MY_CACHE_DIR
    )
    return model, tokenizer, description_tokenizer, device


model, tokenizer, description_tokenizer, device = load_parler()

st.title("🎙️ Generator mowy")
st.markdown("Aplikacja służy do generowania głosu w 8 różnych europejskich językach 🇵🇱 🇬🇧 🇫🇷 🇩🇪 "
            "\n\n\nNajpierw skonfiguruj głos w lewym panelu ⚙️, a następnie wprowadź tekst który ma być przeczytany 🗣️"
            "\n\n\n Po uzyskaniu wystarczającej jakości możesz pobrać nagranie w postaci pliku WAV 💾 "
            )

# --- PANEL BOCZNY (STEROWANIE GŁOSEM) ---
st.sidebar.header("⚙️ Ustawienia głosu")

# Wybór języka i lektora
lang = st.sidebar.selectbox("Język lektora:", list(SPEAKERS.keys()))
speaker = st.sidebar.selectbox("Konkretny głos (Speaker):", SPEAKERS[lang])

# Parametry mowy
speed = st.sidebar.select_slider("Szybkość mowy:",
                                 options=["very slow", "slow", "moderate", "fast", "very fast"],
                                 value="moderate")
pitch = st.sidebar.select_slider("Ton głosu (Pitch):",
                                 options=["very low", "low", "moderate", "high", "very high"],
                                 value="moderate")
expression = st.sidebar.selectbox("Ekspresja:",
                                  ["monotone", "slightly expressive", "animated", "expressive"])
quality = st.sidebar.selectbox("Jakość nagrania:",
                               ["very clear audio", "clear audio", "slightly noisy"])


# --- FUNKCJA BUDUJĄCA OPIS ---
def build_description():
    desc = f"{speaker}'s voice is {expression} with {speed} speed and {pitch} pitch. "
    desc += f"The recording is {quality} with no background noise."
    return desc


generated_description = build_description()

# --- GŁÓWNY INTERFEJS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Tekst")
    prompt = st.text_area("Co lektor ma powiedzieć?",
                          value="Jak Ci się podoba ten głos?",
                          height=150)

    st.info(f"**Wygenerowany opis (Prompt):**\n\n*{generated_description}*")

with col2:
    st.subheader("Generowanie")
    if st.button("🚀 Generuj dźwięk", use_container_width=True):
        if prompt.strip():
            with st.spinner("Model pracuje..."):
                try:
                    # Tokenizacja
                    input_ids = description_tokenizer(generated_description, return_tensors="pt").input_ids.to(device)
                    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                    # Generowanie
                    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                    audio_arr = generation.cpu().numpy().squeeze()

                    # Audio Player
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_arr, model.config.sampling_rate, format='WAV')

                    st.success("Gotowe!")
                    st.audio(buffer, format="audio/wav")

                    # Pobieranie
                    st.download_button(label="💾 Pobierz plik WAV",
                                       data=buffer.getvalue(),
                                       file_name="parler_output.wav",
                                       mime="audio/wav")
                except Exception as e:
                    st.error(f"Wystąpił błąd: {e}")
        else:
            st.warning("Wpisz tekst do przeczytania!")

st.divider()
st.caption("s27794")