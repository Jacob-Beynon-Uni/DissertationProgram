from os import system, name
import os.path
from ImageDetection import imageDetection
from VideoDetection import videoDetection

def clear():
    if name == 'nt': _= system('cls')
    else: _= system('clear')


class program:
    def __init__(self, im, i, o, ob, y, f, c, t):
        self.image_path = im
        self.input_Path = i
        self.output_Path = o
        self.object_Detected = ob
        self.yolo_Path = y
        self.fps = f
        self.confidence = c
        self.threshold = t

    def run_Program(self):
        usr = int(input('1) Image, 2)Video'))
        if usr == 1 : imageDetection(self.image_path, self.yolo_Path, self.confidence, self.threshold)
        elif usr == 2: videoDetection(self.input_Path, self.output_Path, self.yolo_Path, self.object_Detected, self.fps, self.confidence, self.threshold)
        else:
            print("[ERROR] Invalid input")
    
    def help_page(self):
        print('Displaying Help Page')

    def changeSettings(self):
        self.image_path = str(input('Image for detection path:'))
        self.input_Path = str(input("Video Input Path:"))
        self.output_Path = str(input("Video Output Path and name:"))
        self.yolo_Path = str(input("Path to folder containing the yolo files:"))
        self.object_Detected = str(input("Object being detected:"))
        self.fps = int(input("Fps for outputted video:"))
        self.confidence = float(input("Confidence Level:"))
        self.threshold = float(input("New Float Level:"))
        self.writeFile()

    
    def settings(self):
        print('-=-=-=-=- Settings -=-=-=-=-')
        print('-=-=-Paths-=-=-')
        print('Image:' + self.image_path)
        print('Yolo:' + self.yolo_Path)
        print('Input Video:' + self.input_Path)
        print('Output Video:' + self.output_Path)
        print(' ')
        print('-=-=-options-=-=-')
        print('Object being detected: ' + self.object_Detected)
        print('FPS of outputted video: ' + str(self.fps))
        print('Confidence: ' + str(self.confidence))
        print('Threshold ' + str(self.threshold))
        usr_in = str(input(('change or exit settings')))
        if usr_in == "change":
            self.changeSettings()
            

    def history(self):
        print('Displaying History videos')
    
    def main_Menu(self):
        title = 'Object Detection using Visual Information'
        authour = 'By Jacob Beynon'
        options = ['1)Run program', '2)Settings', '3)Help Page', '4)History', '5)Quit']
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print(title.center(45))
        print(authour.center(45))
        print('  Options')
        for x in options:
            print('  ' + x)
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

    def readFile(self):
        print('[INFO] Reading settings from file')
        f = open("Config.txt", "r")
        self.image_path = f.readline()
        self.input_Path = f.readline()
        self.output_Path = f.readline()
        self.yolo_Path = f.readline()
        self.object_Detected = f.readline()
        self.fps = int(f.readline())
        self.confidence = float(f.readline())
        self.threshold = float(f.readline())
        print('[INFO] Closing config file')
        f.close()

    def writeFile(self):
        print('[INFO] Writting Settings to file')
        f = open('Config.txt', 'w')
        f.write(self.image_path)
        f.write('\n')
        f.write(self.input_Path)
        f.write('\n')
        f.write(self.output_Path)
        f.write('\n')
        f.write(self.yolo_Path)
        f.write('\n')
        f.write(self.object_Detected)
        f.write('\n')
        f.write(str(self.fps))
        f.write('\n')
        f.write(str(self.confidence))
        f.write('\n')
        f.write(str(self.threshold))
        print('[INFO] Closing config file')
        f.close()

    def run(self):
        quit = False

        if os.path.exists('Config.txt'): self.readFile()
        else:
            usr = str(input('[ERROR] Config File not found, enter in paths? y, n: '))
            if usr.lower() == 'y': self.changeSettings()

        while quit == False:
            self.main_Menu()
            try: usr_in = int(input('Option:'))
            except: print('Error, invalid input')
            if usr_in == 1: self.run_Program()
            elif usr_in == 2: self.settings()
            elif usr_in == 3: self.help_page(self)
            elif usr_in == 4: self.history(self)
            elif usr_in == 5: quit = True
 
    
        
if __name__ == '__main__':
    p = program('', '', '', 'car', 'YoloFiles', 30, 0.5, 0.3)
    p.run()
