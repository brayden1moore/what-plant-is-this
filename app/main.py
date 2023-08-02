from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os

os.environ['CLIENT_ID'] = '111611815468177630937'
os.environ['CLIENT_EMAIL'] = "429759680026-compute@developer.gserviceaccount.com"
os.environ['PRIVATE_KEY_ID'] = "70bab06cdf7916b8e5ed339a73dd798b2bb96cd5"
os.environ['PRIVATE_KEY'] = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDQEmelgCUZrRBz\nSwFu9E77lw+lFXOGTdCJgl/rZ5IKSSLYYsKGZnj8Yat3DNeBMpHNv5R5FJIoXtIS\nJD3saibGfhmZQFwhHsjoItdK4HEgN9U+9vRlRIN5y/3i9DKbv5M3vCZgagtK9f5C\naimCca8LV8ZQYU/RA4q2wXRso0An89+GTtV3U4qpZaaLHQb4ZN4Qp5Tg/P6YfEnp\njKETLPU9YHzAvNUFuxXnypm5lg2BdY7vHgSMlkCVJ9hMBGquYF+SM61PlezJ/Oqj\nl1AzuaKR8qu8+UAet+OGHGsBXQSilCNkfq9PfIYifYT3o34m8PBij1r4AJ48erUG\nKhkBWMNVAgMBAAECggEAO576eg7lEp1nmFHGwF9a/naDsh8ackJ73dsw1whfbXkV\ndgGekdptEox+EGfqnIe8BcO+rI87bjv1X+NopwSnxbq+ZQ5vF8J1eSb6n+b+I2g8\nP4WN9DKUpeLRBiZJFh3n9lGAgaIBSGKCj89Rw6IFsW9eUQwBTfgA2GtIjBSfPpL7\ndCYHV/B0pHVwel6f4M9G7jYzLj3Azjd16eVnOswA89DfxSB7MqM+My8ZB/mJqrz7\nCSEZ1j0v/Ip6QH6hmlub9B7wD5CzVPxQoi0zccqjukl9a/KxpEm+JamW2m12yoXq\nwUnRbRTwKOUs7MSpUEjko78pt1Gev9M3Y4386jejpQKBgQDxcZAbKf3ZUZZzmuTw\nhW/zEhzbsydrB5oSN7+aiJf+nH3yXg2CoGdEBUElYGnJYuMYrcKVkM0wXVI7MrhS\nR0jRfq0u4I72/ZckR4Y8ZqMRRjw5rGxW8m5ssbrmHS0KH2/P6ULh3cIppUkUzMOD\nLoFFLJDiLi/zqcIi1A06jLwh3wKBgQDcnciQQrtJNRm9sKMK/fZvRmVMJ+s1zcL7\n+vz0b1roNT8+2WQbupdUN46lO4rTAkbAYs8QDrvgvZ1wIFw/ioF1raAzquc9Q1EE\no4B/ts/ZminOPW0cxXQkDJAiZevT9d/4X8+z4pkYZfUQdMtr+4clAZ8inMGFGws/\nbIIYfDYJSwKBgFgSVsCx5pk5O6pb3BsocZe3CbPSfBR8p2Tx1QCnxtnnd8HLMR5v\nKHwVdpgNvUjqu3ArIgmw0khMIkzZyYap3hQdI0swOrY59sITHRI3VlBc0GcxUCu/\nLyyTAFwkVGOW6BBtRCpj3AmY8zmVH9RgSGNVSFxZAMDfMaPGujSbVZz9AoGBAMrG\nZ8tY2qWuHevBSArZZMHgVUkLQ+DfMAHFLu1I0KiwEGKnE6F8/ozUx9LNiIrsA7Xe\n2+0pbbxi7CtcQw3QM6/DF4WF5ybjEbuOwJQipqaeUSCUSw7v2hEsTuqe/YSD8Qls\nnw77DrZjOD2Y7ERjG1ODSw5YQHMkaVExXd49hLlDAoGBAKnow9jIeUiK32co5u95\nppKEBpNBRqLETI/K2ZEfW7eHy+AfjFHTvgJxA6zEEL3xMEw5YRlgiatX/kSt+URU\nrW+hpkknmSPoRdCC1s6Y9f1KUUVwUbKRJKK3JyPvuYU9eLYiZSGeJrtCNipCZLl5\nvBRJ45YoYuK5emVECShgCI6T\n-----END PRIVATE KEY-----\n"

credentials_dict = {
    'type': 'service_account',
    'client_id': os.environ['CLIENT_ID'],
    'client_email': os.environ['CLIENT_EMAIL'],
    'private_key_id': os.environ['PRIVATE_KEY_ID'],
    'private_key': os.environ['PRIVATE_KEY'],
}

credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    credentials_dict
)
client = storage.Client(credentials=credentials, project='bmllc-plant')
bucket = client.get_bucket('bmllc-plant-image-bucket')

import plantvision
import pickle as pkl
from flask import Flask, render_template, request, session, jsonify, url_for
from PIL import Image
import os
import random
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

app = Flask(__name__)
app.secret_key = 'pi-33pp-co-sk-33'
app.template_folder = os.path.abspath(f'{THIS_FOLDER}/web/templates')
app.static_folder = os.path.abspath(f'{THIS_FOLDER}/web/static')
print(app.static_folder)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guess', methods=['POST'])
def guess():

    session['sessionId'] = random.random()*100
    if request.method == 'POST':
        print('Thinking...')
        img = request.files.get('uploaded-image')
        feature = request.form.get('feature')

        tensor = plantvision.processImage(img, feature)
        predictions = plantvision.see(tensor, feature, 9)
        #confidences = [f'{str(round(i*100,4))}%' for i in confidences]

        with open(f'{THIS_FOLDER}/resources/speciesNameToKey.pkl','rb') as f:
            speciesNameToKey = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/speciesNameToVernacular.pkl','rb') as f:
            speciesNameToVernacular = pkl.load(f)
        with open(f'{THIS_FOLDER}/resources/{feature}speciesIndexDict.pkl','rb') as f:
            speciesNameToIndex = pkl.load(f)

        urls = []
        predictedImages = []
        for p in predictions:
            key = speciesNameToKey[p]
            img = speciesNameToIndex[p]
            query = ''
            for i in p.split(' '):
                query += i 
                query += '+'
            urls.append(f'https://www.google.com/search?q={query[:-1]}')
            predictedImages.append(f'{THIS_FOLDER}/images/img{img}.jpeg')

        predicted_image_urls = []
        for i,image in enumerate(predictedImages):
            blob = bucket.blob(f"{session['sessionId']}_{i}.jpeg")
            blob.upload_from_filename(image)
            predicted_image_urls.append(f"https://storage.cloud.google.com/bmllc-plant-image-bucket/{session['sessionId']}_{i}.jpeg")

        names = []
        for p in predictions:
            try:
                names.append(speciesNameToVernacular[p])
            except:
                names.append(p)

        response = {
            'names': names,
            'species': predictions,
            'predictions': urls,
            'images': predicted_image_urls
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run()
    #app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)