# download_models.py
import urllib.request

urls = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/age_deploy.prototxt",
    "age_net.caffemodel": "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/age_net.caffemodel",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_net.caffemodel"
}

for fname, url in urls.items():
    print(f"Downloading {fname} …")
    urllib.request.urlretrieve(url, fname)
print("All done — models are in this folder.")


