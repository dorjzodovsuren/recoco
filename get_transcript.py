import os
import io
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./key/top-campaign-313812-50890e9f72d0.json"

def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="ja-JP",  # en-US
        audio_channel_count=2,
        enable_word_time_offsets=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    transcript = []
    response = operation.result()

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        inst_transcript = result.alternatives[0].transcript
        transcript.append(inst_transcript)
    
    return " ".join(transcript).strip()