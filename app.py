import os
import sys
import time
import requests
import assemblyai
import moviepy.editor as mp
from get_transcript import transcribe_gcs
from upload_audio import upload_blob
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, redirect, flash

# Authentication 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/dorjzodovs.batjargal/Desktop/Personal/Speech2Text/Speech2Text_backend/key/top-campaign-313812-50890e9f72d0.json"


# print(transcribe_gcs("gs://inputaudio_website/The Speech that Made Obama President.wav"))


# bucket_name="inputaudio_website"
# source_file_name="./saver/MIB2.wav"
# destination_blob_name="MIB2.wav"

# upload_blob(bucket_name,source_file_name,destination_blob_name)


#-----------------------Start of Flask app settings------------------------------------------------------------------------

UPLOAD_FOLDER = './saver'            
              
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'wav', 'mp3', 'webm', 'wmv', 'flv'} 

TRANSCRIPT_SAVE_DIR= './transcript/'


PORT=5000

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'                           

with open('token.txt') as token_file:
    YOUR_API_TOKEN = token_file.read()

db = SQLAlchemy(app)

#-----------------------End of Flask app settings------------------------------------------------------------------------

def allowed_file(filename,ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def audio_saver(filename,video_path):
    if allowed_file(filename,{'wav', 'mp3'}):
        return filename

    else:

        filename_root=filename.rsplit('.', 1)[0]
        my_clip = mp.VideoFileClip(r"%s"%video_path)
        my_clip.audio.write_audiofile(r"%s/%s.wav"%(UPLOAD_FOLDER,filename_root))
        return "%s.wav"%filename_root
 
def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data
 

def preprocess_text(text):
    processing_text=text.replace("?",".")
    processing_text=processing_text.replace("!",".")
    return processing_text


import assemblyai

#need to be replaced
def audio2text(URL,YOUR_API_TOKEN):

    aai = assemblyai.Client(token=YOUR_API_TOKEN)

    transcript = aai.transcribe(audio_url=URL,speaker_count=1)

    while transcript.status != 'completed':
        task_content_func = transcript.get()
            
    task_content=task_content_func.text

    return task_content


class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(1000), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.now())
    
    def __repr__(self):
        return '<Task %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        task_name = request.form['content']

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename,ALLOWED_EXTENSIONS):

            filename = secure_filename(file.filename)

            destination_blob_name=filename

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            video_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filename=audio_saver(filename,video_path)           #modified name of the file
            filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)

            print(destination_blob_name)

            # bucket_name="inputaudio_website"
            # source_file_name="./saver/MIB2.wav"
            # destination_blob_name="MIB2.wav"

            print(filename)

            # upload_blob(bucket_name,source_file_name,destination_blob_name)

            headers = {'authorization': YOUR_API_TOKEN}
            response = requests.post('https://api.assemblyai.com/v2/upload',
                                    headers=headers,
                                    data=read_file(filename))

            URL=response.json()["upload_url"]

        task_content=audio2text(URL,YOUR_API_TOKEN)

        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)

            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your task'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        if len(tasks)>0:
            last_task=[tasks[-1]]
        else:
            last_task=tasks
        return render_template('index.html', tasks=last_task)


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that task'


@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your task'

    else:
        return render_template('update.html', task=task)


@app.route('/download/<int:id>')
def download(id):
    task_to_download = Todo.query.get_or_404(id)
    content=task_to_download.content

    text_file = open(TRANSCRIPT_SAVE_DIR+"%s.txt"%str(id), "w")
    n = text_file.write(content)
    text_file.close()
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
