# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.

# use default Django user model to store basic user information (username/pass, email, name, etc.)
# make profile model w/ 1-1 mapping to user model based on id, where profile includes nutrtion preferences/stats