import streamlit as st
from pathlib import Path
import numpy as np
import traceback
import os

try:
    from inference.tts_engine import TTSEngine, TTSConfig
except ImportError as e:
    st.error(f"""
    **Failed to import TTSEngine.**

    *   Ensure the directory structure is correct:
        ```
        your_project_root/
        ├── inference/
        │   ├── __init__.py
        │   ├── infer.py
        │   └── tts_engine.py
        ├── ui/
        │   ├── __init__.py
        │   └── app.py
        ├── main.py
        └── ... (other files/dirs)
        ```
    *   Make sure empty `__init__.py` files exist in `inference/` and `ui/`.
    *   Run Streamlit from `your_project_root`.
    *   Original Error: `{e}`
    """)
    st.stop()


MULTI_SPEAKER_MODEL = "tts_models/en/vctk/vits"


CONFIG = TTSConfig(
    model_name=MULTI_SPEAKER_MODEL,
    model_path=None,
    embeddings_path=None
)

@st.cache_resource
def get_engine_and_speakers():
    """Loads the engine and gets the list of the first 50 built-in speaker names."""
    print(f"INFO: Loading TTS Engine with model: {CONFIG.model_name}")

    engine = None
    speaker_name_list = ["Default Model Voice"]

    spinner_message = f"Initializing TTS Engine ({CONFIG.model_name})... Please wait."

    with st.spinner(spinner_message):
        try:
            engine = TTSEngine(CONFIG)

            if engine and hasattr(engine, 'speakers') and engine.speakers:
                 model_speakers = engine.speakers
                 if isinstance(model_speakers, list) and all(isinstance(s, str) for s in model_speakers):
                      total_speakers = len(model_speakers)
                      print(f"INFO: Model '{CONFIG.model_name}' has {total_speakers} built-in speakers.")

                      speakers_to_show = sorted(model_speakers)[:50]

                      speaker_name_list.extend(speakers_to_show)
                      print(f"INFO: Populated dropdown with first {len(speakers_to_show)} speakers (out of {total_speakers} total).")
                 else:
                      st.warning(f"Model has '.speakers' attribute, but it's not a list of strings (Type: {type(model_speakers)}). Cannot populate speaker list.")
            elif engine:
                 st.warning(f"Loaded model '{CONFIG.model_name}' does not seem to have a built-in speaker list (`.speakers` attribute missing or empty). Using default voice only.")

            return engine, speaker_name_list

        except Exception as e:
            st.error(f"Engine initialization failed: {str(e)}")
            st.text(traceback.format_exc())
            return None, ["Default Model Voice"]


def main():
    st.title("🎤 Multi-Speaker TTS System")

    engine, speaker_name_list = get_engine_and_speakers()

    speaker_selection_disabled = (not engine or len(speaker_name_list) <= 1)

    selected_speaker_name = st.selectbox(
        "Select Speaker Voice",
        options=speaker_name_list,
        index=0,
        disabled=speaker_selection_disabled,
        help="Select a speaker name from the model's internal list (display limited to 50)."
    )

    speaker_id_to_pass = None
    if selected_speaker_name != "Default Model Voice":
        speaker_id_to_pass = selected_speaker_name

    text = st.text_area("Input Text", "Enter text for voice synthesis.")

    if st.button("Generate Audio"):
        if not engine:
            st.error("TTS Engine not available. Cannot generate audio.")
            st.stop()
        if not text.strip():
            st.warning("Please enter some text to synthesize.")
            st.stop()

        voice_info = f"Voice: {selected_speaker_name}" if speaker_id_to_pass else "Voice: Default Model Voice"
        with st.spinner(f"Generating audio... ({voice_info})"):
            try:
                audio_data = engine.synthesize(text, speaker_id=speaker_id_to_pass)

                if audio_data is not None and isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                    output_filename = "output.wav"
                    engine.save_to_wav(audio_data, output_filename)
                    output_file = Path(output_filename)
                    if output_file.exists() and output_file.stat().st_size > 44:
                         st.audio(output_filename)
                         st.success("Generation complete!")
                    else:
                         st.error(f"Output file '{output_filename}' missing or empty after saving.")
                else:
                    st.error("Generation failed. Synthesize function returned invalid data. Check console logs.")

            except Exception as e:
                st.error(f"An error occurred during synthesis or saving: {str(e)}")
                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()