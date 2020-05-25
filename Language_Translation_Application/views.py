from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from Language_Translation import process
from django.http import HttpResponse
# Create your views here.

@csrf_exempt
def getResult(request):
    if request.method == 'GET':
        return render(request, "index.html")
    if request.method == 'POST':
        text = request.POST.get('lan')
        response = process.return_sentences(text)
        print(response)
        return HttpResponse(response, status=200)

