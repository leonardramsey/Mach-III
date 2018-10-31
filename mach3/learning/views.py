# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.


def index(request):
    if request.method == 'POST':
        return JsonResponse({'message':'Hello World'})
    return render(request, 'general/index.html')