import speech_recognition as sr 
import pyttsx3 as ptt
from datetime import datetime

robot_ear = sr.Recognizer() 
robot_mounth = ptt.init()
you = ""
robot_brain = "I can't hear you voice."

def _listen():
    global you
    with sr.Microphone() as mic:
        print("Robot: I'm listening.")
        audio = robot_ear.listen(mic)
    try:
        you = robot_ear.recognize_google(audio)
    except:
        you = "" 
    print("You: " + you)    

def _thingking():
    global robot_brain
    print("Robot: ...")
    if you == "":
        robot_brain = "I can't hear you, try again"
    elif "hello" in you:
        robot_brain = "Hello Tuan" 
    elif "today" in you: 
        robot_brain = "today is " + datetime.today().strftime('%Y-%m-%d') 
    elif "time" in you: 
        robot_brain = "the time is " + datetime.today().strftime('%H:%M') 
    elif "bye" in you:
        robot_brain = "Good bye"
        robot_mounth.say(robot_brain)
        robot_mounth.runAndWait()
        exit()
    else:
        robot_brain = "I cannot understand what you say, Please try again."
    print("Robot: " + robot_brain)

def _answer(): 
    robot_mounth.say(robot_brain)
    robot_mounth.runAndWait()

# Main process
while True:
    _listen()
    _thingking()
    _answer()