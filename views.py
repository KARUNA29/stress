from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from .models import UserRegistrationModel, UserImagePredictinModel
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .utility.GetImageStressDetection import ImageExpressionDetect
from .utility.MyClassifier import KNNclassifier
from subprocess import Popen, PIPE
import subprocess
import sqlite3
import io
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # To render plots without GUI

# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def UploadImageForm(request):
    loginid = request.session['loginid']
    data = UserImagePredictinModel.objects.filter(loginid=loginid)
    return render(request, 'users/UserImageUploadForm.html', {'data': data})


def UploadImageAction(request):
    image_file = request.FILES['file']

    # let's check if it is a csv file
    if not image_file.name.endswith('.jpg'):
        messages.error(request, 'THIS IS NOT A JPG  FILE')

    fs = FileSystemStorage()
    filename = fs.save(image_file.name, image_file)
    # detect_filename = fs.save(image_file.name, image_file)
    uploaded_file_url = fs.url(filename)
    obj = ImageExpressionDetect()
    emotion = obj.getExpression(filename)
    username = request.session['loggeduser']
    loginid = request.session['loginid']
    email = request.session['email']
    UserImagePredictinModel.objects.create(username=username, email=email, loginid=loginid, filename=filename,
                                           emotions=emotion, file=uploaded_file_url)
    data = UserImagePredictinModel.objects.filter(loginid=loginid)
    return render(request, 'users/UserImageUploadForm.html', {'data': data})


def UserEmotionsDetect(request):
    if request.method == 'GET':
        imgname = request.GET.get('imgname')
        obj = ImageExpressionDetect()
        emotion = obj.getExpression(imgname)
        loginid = request.session['loginid']
        data = UserImagePredictinModel.objects.filter(loginid=loginid)
        return render(request, 'users/UserImageUploadForm.html', {'data': data})


def UserLiveCameDetect(request):
    obj = ImageExpressionDetect()
    obj.getLiveDetect()
    return render(request, 'users/UserLiveHome.html', {})


def UserKerasModel(request):
    # p = Popen(["python", "kerasmodel.py --mode display"], cwd='StressDetection', stdout=PIPE, stderr=PIPE)
    # out, err = p.communicate()
    subprocess.call("python kerasmodel.py --mode display")
    return render(request, 'users/UserLiveHome.html', {})


import sqlite3
import io
import base64
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.contrib import messages
from .utility.GetImageStressDetection import ImageExpressionDetect
from users.models import UserImageEmotions  # Assuming the model name is UserImageEmotions

def UserKNNResults(request):
    try:
        # Get the current user's login ID
        login_id = request.session.get('loginid')

        if login_id:
            # Connect to the SQLite database
            conn = sqlite3.connect('C:/Users/karun/OneDrive/Desktop/Stress_Detection_Final_Model/db.sqlite3')
            cursor = conn.cursor()

            # Retrieve the username and emotions data for the current user from the table
            cursor.execute('SELECT username, emotions FROM UserImageEmotions WHERE loginid = ?', (login_id,))
            results = cursor.fetchall()

            if results:
                # Process the retrieved data
                emotions = [row[1] for row in results]

                # Count the occurrences of each emotion category
                emotion_counts = {}
                for emotion in emotions:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
                    else:
                        emotion_counts[emotion] = 1

                # Create a pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=140)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title('Distribution of Emotions')

                # Save the pie chart to a BytesIO object
                pie_chart_buffer = io.BytesIO()
                plt.savefig(pie_chart_buffer, format='png')
                plt.close()

                # Set the buffer's pointer to the beginning
                pie_chart_buffer.seek(0)

                # Encode the plot image to a base64 string
                pie_chart_base64 = base64.b64encode(pie_chart_buffer.getvalue()).decode('utf-8')

                # Render the template with the image URL
                context = {'pie_chart_image': pie_chart_base64}
                return render(request, 'users/UserKnnResults.html', context)
            else:
                messages.info(request, 'No emotions data found for the current user.')
                return render(request, 'users/UserKnnResults.html', {})

        else:
            messages.error(request, 'User not logged in.')
            return render(request, 'users/UserLogin.html', {})

    except Exception as e:
        # Handle exceptions
        messages.error(request, f"An error occurred: {str(e)}")
        return render(request, 'users/UserKnnResults.html', {})
