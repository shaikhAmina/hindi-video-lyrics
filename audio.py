import os
import re
import time
import tempfile
import subprocess

import streamlit as st
from faster_whisper import WhisperModel
from fpdf import FPDF


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Video to Lyrics PDF",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Video → Lyrics / Transcript PDF")

st.write(
    "Upload a video, select the language and audio type, "
    "then generate a transcript or lyrics PDF."
)


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


# ==========================================================
# LANGUAGE OPTIONS
# ==========================================================

LANGUAGES = {
    "Hindi": "hi",
    "English": "en",
    "Marathi": "mr",
    "Urdu": "ur",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Nepali": "ne",
    "Auto Detect": None,
}


# ==========================================================
# AUDIO TYPE
# ==========================================================

AUDIO_TYPES = {
    "🎵 Song / Lyrics": "song",
    "🗣️ Normal Speech": "speech",
}


# ==========================================================
# SESSION STATE
# ==========================================================

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "segments" not in st.session_state:
    st.session_state.segments = []

if "language" not in st.session_state:
    st.session_state.language = ""

if "language_probability" not in st.session_state:
    st.session_state.language_probability = 0.0

if "processing_done" not in st.session_state:
    st.session_state.processing_done = False


# ==========================================================
# LOAD WHISPER MODEL
# ==========================================================

@st.cache_resource
def load_model():

    start_time = time.time()

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    elapsed = time.time() - start_time

    return model, elapsed


# ==========================================================
# CHECK FFMPEG
# ==========================================================

def check_ffmpeg():

    try:

        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        return result.returncode == 0

    except FileNotFoundError:

        return False


# ==========================================================
# EXTRACT AUDIO
# ==========================================================

def extract_audio(video_path):

    audio_path = os.path.join(
        tempfile.gettempdir(),
        "whisper_input_audio.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,

        # Mono audio
        "-ac",
        "1",

        # Whisper uses 16kHz audio
        "-ar",
        "16000",

        # PCM WAV
        "-c:a",
        "pcm_s16le",

        audio_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg audio extraction failed:\n\n"
            + result.stderr[-3000:]
        )

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            "Audio file was not created."
        )

    return audio_path


# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_text(text):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ==========================================================
# REMOVE IMMEDIATE REPEATED PHRASES
# ==========================================================

def remove_repeated_phrases(text):

    words = text.split()

    if len(words) < 4:
        return text

    result = []

    i = 0

    while i < len(words):

        removed = False

        # Check 5-word, 4-word, 3-word,
        # 2-word and 1-word repetitions
        for phrase_length in range(
            min(5, len(words) // 2),
            0,
            -1
        ):

            if (
                i + phrase_length * 2
                <= len(words)
            ):

                first = words[
                    i:i + phrase_length
                ]

                second = words[
                    i + phrase_length:
                    i + phrase_length * 2
                ]

                if first == second:

                    result.extend(first)

                    i += phrase_length * 2

                    removed = True

                    break

        if not removed:

            result.append(words[i])

            i += 1

    return " ".join(result)


# ==========================================================
# DETECT EXTREME REPETITION
# ==========================================================

def is_excessively_repeated(text):

    words = text.split()

    if len(words) < 8:
        return False

    # Example:
    #
    # hello hello hello hello hello
    #
    unique_words = set(
        word.lower()
        for word in words
    )

    if (
        len(unique_words) <= 3
        and len(words) >= 8
    ):
        return True

    # Check repeated chunks
    for size in range(1, 6):

        if len(words) < size * 4:
            continue

        chunks = []

        for i in range(
            0,
            len(words) - size + 1
        ):

            chunk = tuple(
                word.lower()
                for word in words[
                    i:i + size
                ]
            )

            chunks.append(chunk)

        counts = {}

        for chunk in chunks:

            counts[chunk] = (
                counts.get(chunk, 0) + 1
            )

        if counts:

            highest = max(
                counts.values()
            )

            if highest >= 4:
                return True

    return False


# ==========================================================
# CLEAN SEGMENTS
# ==========================================================

def clean_segments(raw_segments):

    cleaned = []

    previous_text = ""

    for segment in raw_segments:

        text = normalize_text(
            segment["text"]
        )

        if not text:
            continue

        # Remove immediate repetitions
        text = remove_repeated_phrases(
            text
        )

        # Skip extreme hallucination
        if is_excessively_repeated(text):
            continue

        # Skip exact duplicate consecutive segment
        if (
            text.lower()
            == previous_text.lower()
        ):
            continue

        cleaned.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": text
        })

        previous_text = text

    return cleaned


# ==========================================================
# FORMAT TIME
# ==========================================================

def format_time(seconds):

    minutes = int(seconds // 60)

    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


# ==========================================================
# CREATE TRANSCRIPT TEXT
# ==========================================================

def segments_to_text(segments):

    return "\n\n".join(
        segment["text"]
        for segment in segments
        if segment["text"]
    )


# ==========================================================
# HINDI CHARACTER RATIO
# ==========================================================

def hindi_ratio(text):

    if not text:
        return 0

    hindi = 0
    total = 0

    for char in text:

        if char.isspace():
            continue

        total += 1

        if "\u0900" <= char <= "\u097F":
            hindi += 1

    if total == 0:
        return 0

    return hindi / total


# ==========================================================
# CREATE PDF
# ==========================================================

def create_pdf(text, output_path):

    font_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "fonts",
        "NotoSansDevanagari-Regular.ttf"
    )

    if not os.path.exists(font_path):

        raise FileNotFoundError(
            f"""
Hindi/Unicode font not found.

Expected:

{font_path}

Please put:

NotoSansDevanagari-Regular.ttf

in the same folder as audio.py.
"""
        )

    pdf = FPDF()

    pdf.add_page()

    pdf.add_font(
        "NotoDevanagari",
        "",
        font_path
    )

    pdf.set_font(
        "NotoDevanagari",
        size=14
    )

    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )

    pdf.multi_cell(
        0,
        10,
        text
    )

    pdf.output(output_path)


# ==========================================================
# HEADER
# ==========================================================

st.subheader("📤 Upload Video")


# ==========================================================
# UPLOAD VIDEO
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose any video file",
    type=[
        "mp4",
        "mov",
        "mkv",
        "avi",
        "webm",
        "mpeg",
        "mpg"
    ]
)


# ==========================================================
# SETTINGS UI
# ==========================================================

if uploaded_file:

    st.subheader("⚙️ Transcription Settings")

    col1, col2 = st.columns(2)

    with col1:

        language_name = st.selectbox(
            "🌐 Language",
            options=list(
                LANGUAGES.keys()
            ),
            index=0
        )

    with col2:

        audio_type_name = st.selectbox(
            "🎵 Audio Type",
            options=list(
                AUDIO_TYPES.keys()
            ),
            index=0
        )

    selected_language = LANGUAGES[
        language_name
    ]

    audio_type = AUDIO_TYPES[
        audio_type_name
    ]

    st.caption(
        "For your Hindi songs, select "
        "**Hindi + Song / Lyrics**."
    )


    # ======================================================
    # SAVE VIDEO
    # ======================================================

    video_path = os.path.join(
        tempfile.gettempdir(),
        uploaded_file.name
    )

    with open(
        video_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # ======================================================
    # FILE INFORMATION
    # ======================================================

    file_size_mb = (
        os.path.getsize(video_path)
        / (1024 * 1024)
    )

    st.success(
        f"✅ Video uploaded: "
        f"{uploaded_file.name}"
    )

    st.write(
        f"**Size:** {file_size_mb:.2f} MB"
    )

    st.video(video_path)


    # ======================================================
    # FFMPEG CHECK
    # ======================================================

    ffmpeg_available = check_ffmpeg()

    if not ffmpeg_available:

        st.error(
            """
❌ FFmpeg is not installed.

Install it from Command Prompt:

winget install Gyan.FFmpeg

Then close and reopen your terminal.
"""
        )

        st.stop()


    # ======================================================
    # GENERATE BUTTON
    # ======================================================

    if st.button(
        "🎤 Generate Lyrics / Transcript",
        type="primary"
    ):

        try:

            # ==================================================
            # CLEAR OLD RESULT
            # ==================================================

            st.session_state.transcript = ""

            st.session_state.segments = []

            st.session_state.processing_done = False


            # ==================================================
            # STEP 1
            # ==================================================

            st.subheader(
                "1️⃣ Loading Whisper model"
            )

            model_start = time.time()

            model, model_load_time = (
                load_model()
            )

            model_time = (
                time.time()
                - model_start
            )

            st.success(
                f"✅ Whisper model ready "
                f"in {model_time:.2f} seconds"
            )


            # ==================================================
            # STEP 2
            # ==================================================

            st.subheader(
                "2️⃣ Extracting audio"
            )

            audio_start = time.time()

            with st.spinner(
                "🎧 Extracting audio from video..."
            ):

                audio_path = extract_audio(
                    video_path
                )

            audio_time = (
                time.time()
                - audio_start
            )

            st.success(
                f"✅ Audio extracted in "
                f"{audio_time:.2f} seconds"
            )


            # ==================================================
            # STEP 3
            # ==================================================

            st.subheader(
                "3️⃣ Transcribing"
            )

            transcription_start = (
                time.time()
            )

            status = st.empty()

            status.info(
                "🎤 Whisper is processing the audio..."
            )


            # ==================================================
            # TRANSCRIPTION SETTINGS
            # ==================================================

            if audio_type == "song":

                # Songs can confuse VAD because
                # music is not normal speech.
                vad_filter = False

                # Avoid carrying bad lyrics from one
                # musical segment into the next.
                condition_on_previous_text = False

            else:

                # Normal speech
                vad_filter = True

                condition_on_previous_text = True


            # ==================================================
            # WHISPER
            # ==================================================

            segments, info = model.transcribe(

                audio_path,

                # ----------------------------------------------
                # IMPORTANT
                # ----------------------------------------------
                #
                # If Hindi is selected:
                #
                #     language = "hi"
                #
                # If Auto Detect:
                #
                #     language = None
                #
                language=selected_language,

                task="transcribe",

                # Better decoding
                beam_size=5,

                best_of=5,

                # Different behavior for songs/speech
                condition_on_previous_text=(
                    condition_on_previous_text
                ),

                temperature=0,

                # VAD depends on audio type
                vad_filter=vad_filter,

                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 300
                } if vad_filter else None,

                # Reduce hallucination
                no_speech_threshold=0.6,

                # Remove very unlikely segments
                log_prob_threshold=-1.0,

                # Detect suspicious repetitive output
                compression_ratio_threshold=2.4
            )


            # ==================================================
            # COLLECT SEGMENTS
            # ==================================================

            raw_segments = []

            segment_count = 0

            for segment in segments:

                segment_count += 1

                text = segment.text.strip()

                if text:

                    raw_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": text
                    })

                    status.info(
                        f"🎤 Processing segment "
                        f"{segment_count}..."
                    )


            transcription_time = (
                time.time()
                - transcription_start
            )


            # ==================================================
            # CLEAN
            # ==================================================

            cleaned = clean_segments(
                raw_segments
            )

            transcript = segments_to_text(
                cleaned
            )


            # ==================================================
            # SAVE RESULT TO SESSION
            # ==================================================

            st.session_state.transcript = (
                transcript
            )

            st.session_state.segments = (
                cleaned
            )

            st.session_state.language = (
                info.language
            )

            st.session_state.language_probability = (
                info.language_probability
            )

            st.session_state.processing_done = True


            # ==================================================
            # RESULT
            # ==================================================

            st.success(
                f"""
✅ Transcription completed!

⏱ Processing time:
{transcription_time:.2f} seconds

📝 Original segments:
{len(raw_segments)}

🧹 Cleaned segments:
{len(cleaned)}

🌐 Detected language:
{info.language}

🎯 Language probability:
{info.language_probability:.2%}
"""
            )


            # ==================================================
            # NO RESULT
            # ==================================================

            if not transcript:

                st.warning(
                    """
⚠️ No usable speech/lyrics were detected.

Possible reasons:

• Music is too loud
• Voice is too quiet
• Audio quality is poor
• Video is mostly instrumental
• Speech is unclear
"""
                )

                st.stop()


        except Exception as e:

            st.error(
                "❌ Error while processing video"
            )

            st.exception(e)


# ==========================================================
# DISPLAY RESULT
# ==========================================================

if (
    st.session_state.processing_done
    and st.session_state.transcript
):

    # ======================================================
    # LANGUAGE RESULT
    # ======================================================

    detected_language = (
        st.session_state.language
    )

    probability = (
        st.session_state.language_probability
    )

    if detected_language == "hi":

        st.success(
            f"🇮🇳 Hindi detected "
            f"({probability:.2%})"
        )

    else:

        st.info(
            f"🌐 Detected language: "
            f"{detected_language} "
            f"({probability:.2%})"
        )


    # ======================================================
    # TIMESTAMPED TRANSCRIPT
    # ======================================================

    st.subheader(
        "⏱ Transcript"
    )

    for segment in (
        st.session_state.segments
    ):

        start = format_time(
            segment["start"]
        )

        end = format_time(
            segment["end"]
        )

        st.write(
            f"**[{start} → {end}]** "
            f"{segment['text']}"
        )


    # ======================================================
    # FINAL TEXT
    # ======================================================

    st.subheader(
        "📝 Final Lyrics / Transcript"
    )

    edited_text = st.text_area(
        "Correct the text if required:",
        value=st.session_state.transcript,
        height=450,
        key="edited_transcript"
    )


    # ======================================================
    # CREATE PDF
    # ======================================================

    if st.button(
        "📄 Create PDF"
    ):

        try:

            pdf_start = time.time()

            pdf_path = os.path.join(
                tempfile.gettempdir(),
                "lyrics_transcript.pdf"
            )

            create_pdf(
                edited_text,
                pdf_path
            )

            pdf_time = (
                time.time()
                - pdf_start
            )

            st.success(
                f"✅ PDF created in "
                f"{pdf_time:.2f} seconds"
            )


            # ==================================================
            # DOWNLOAD
            # ==================================================

            with open(
                pdf_path,
                "rb"
            ) as pdf_file:

                pdf_data = pdf_file.read()


            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_data,
                file_name="lyrics_transcript.pdf",
                mime="application/pdf"
            )


        except Exception as e:

            st.error(
                "❌ Error while creating PDF"
            )

            st.exception(e)