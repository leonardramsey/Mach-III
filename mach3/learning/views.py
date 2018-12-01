# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
import base64, os, urllib, requests

# Create your views here.

ratings = {2: 'GOOD',1: 'OK',0: 'BAD'}

def index(request):
    message=""
    if request.method == 'POST':
        # hit AWS API w/ requests
        label_img = request.POST.get("label_img", "")
        screenshot_img = request.POST.get("screenshot_img_input", "")
        message = 'Image invalid. Please upload an image of a nutrition label.'
        if screenshot_img:
            source = screenshot_img
            with open('../Tesseract/images/image.jpg', 'wb') as img:
                screenshot_img = screenshot_img.partition(',')[2]
                padding= len(screenshot_img) % 4
                screenshot_img += b"=" * padding
                img.write(base64.b64decode(screenshot_img))
            payload = {'label_img': label_img, 'user': request.user}
            # response = requests.post('https://httpbin.org/get', params=payload)
            # check response
            message = 'Image upload successful.'
            rating = ratings[2]
        elif label_img:
            source = label_img
            payload = {'label_img': label_img, 'user': request.user}
            with open("./images/img.jpg", 'w+'):
                pass
            # response = requests.post('https://httpbin.org/get', params=payload)
            # check response
            message = 'Image upload successful.'
            rating = ratings[2]
        else:
            return render(request, 'general/index.html', {'message':message})
        return render(request, 'general/result.html', {'rating': rating, 'source':source})
    return render(request, 'general/index.html', {'message': message})

def result(request):
    message=""
    return render(request, 'general/index.html', {'message': message})