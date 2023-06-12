from django.shortcuts import render
from django.http import HttpResponse

from .actions.train import BotModel
from .actions.predict import BotPredict
# Create your views here.


def index(request):
    return render(request, 'index.html', {'name': 'John Doe'})

def train(request):
    model = BotModel()
    model.train()
    return HttpResponse(f"model predicted...")

def predict(request):
    bot = BotPredict()
    print(bot.predict("hi there"))
    return HttpResponse(f"Training completed...")