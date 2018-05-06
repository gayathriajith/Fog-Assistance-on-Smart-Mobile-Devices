import os
from flask import Flask, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import math
import cv2
import requests
import numpy as np

UPLOAD_FOLDER = 'G:\\college\\FinalYearProject\\34_Fog_Removal\\main'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
    indices = np.argsort(darkvec,0)
    indices = indices[imsz-numpx::]
    b,g,r=cv2.split(im)
    gray_im=r*0.299 + g*0.587 + b*0.114
    gray_im=gray_im.reshape(imsz,1)
    loc=np.where(gray_im==max(gray_im[indices]))
    x=loc[0][0]
    A=np.array(imvec[x])
    A=A.reshape(1,3)
    return A


def TransmissionEstimate(im,A,sz):
    omega = 1
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):

    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))

    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 50
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res

@app.route('/fog', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print 'post'
        # check if the post request has the file part
        if 'file' not in request.files:
            print 'No file part'
            return 'no file sent'
        file = request.files['file']
        if file.filename == '':
            print 'No selected file'
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            try:
                filePath = 'fog.png'
                # img = cv2.flip(img,1)
                src = cv2.imread(filePath)
                h,w,d = src.shape
                print h,w,d
                w1 = int(w/10)
                h1 = int(h/10)
                src = cv2.resize(src,(w1,h1))
            except:
                filePath = 'gallery.png'
                src = cv2.imread(filePath)
                h,w,d = src.shape
                print h,w,d

            I = src.astype('float64')/255
            dark = DarkChannel(I,15)
            A = AtmLight(I,dark)
            te = TransmissionEstimate(I,A,15)
            t = TransmissionRefine(src,te)
            J = Recover(I,t,A,0.1)
            J[J<0]=0
            cv2.imwrite('edit.png',np.uint8(255*J))
            os.remove(filePath)
            return send_file('edit.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
