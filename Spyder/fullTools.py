import os
from time import sleep
from PIL import Image
import xml.etree.ElementTree as ET
from aip import AipImageClassify
from xmlTools import *
from threading import Thread


APP_ID = '23690363'
API_KEY = 'gKtquDgLHfbkqfu2LVeAjXGQ'
SECRET_KEY = 'k72l66w8KCe8gGBtvaGGGy35EDi4i5PF'

FOLDER_PATH = '/home/whut/TensorFlow/workspace/traffic/data/UA-DETRAC/trainpick/'


client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)


def filesRename(folderPath, addName):
    fileList = getFileList(folderPath)
    for fileName in fileList:
        os.rename(folderPath+fileName, folderPath+addName+fileName)


def getFileList(path):
    for a, b, file in os.walk(path):
        return file


def getFolderList(path):
    for a, folder, c in os.walk(path):
        return folder


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def BaiduGetResponse(FOLDER_PATH, fileName):
    image = get_file_content(FOLDER_PATH+fileName)
    response = client.vehicleDetect(image)
    BaiduWriteIntoXML(fileName, FOLDER_PATH, response)


def MoveDoneFiles(FOLDER_PATH):
    fileList = getFileList(FOLDER_PATH)
    fileList.sort()
    for fileName in fileList:
        fileNameWithout = fileName.split('.')[0]
        if(os.path.exists(FOLDER_PATH+fileNameWithout+'.xml') and os.path.exists(FOLDER_PATH+fileNameWithout+'.xml')):
            os.system('mv {0}{1} {0}done/'.format(FOLDER_PATH, fileNameWithout + '.xml'))
            os.system('mv {0}{1} {0}done/'.format(FOLDER_PATH, fileNameWithout + '.jpg'))
            print('{0} moved.'.format(fileNameWithout))
    print('All moved.')


if __name__ == '__main__':
    MoveDoneFiles(FOLDER_PATH)
    fileList = getFileList(FOLDER_PATH)
    try:
        for fileName in fileList:
            Thread(target=BaiduGetResponse, args=(FOLDER_PATH, fileName,)).start()
            sleep(0.1)
    except Exception as e:
        MoveDoneFiles(FOLDER_PATH)
        print(repr(e))

'''
    fileList = getFileList(FOLDER_PATH)
    for fileName in fileList:
        image = get_file_content(FOLDER_PATH+fileName)
        response = client.vehicleDetect(image)
        writeIntoXML(fileName, FOLDER_PATH, response)
        print(fileName)

    os.system('cd {} && mkdir pick'.format(FOLDER_PATH))
    fileList = getFileList(FOLDER_PATH)
    fileList.sort()
    index = 0
    for i in range(len(fileList)):
        os.system('cd {} && mv {} ./pick/'.format(FOLDER_PATH, fileList[index]))
        index += 6
        print(fileList[index])

    folderList = getFolderList(FOLDER_PATH)
    for folderName in folderList:
        fileList = getFileList(FOLDER_PATH+folderName)
        for fileName in fileList:
            os.rename(FOLDER_PATH+folderName+'/'+fileName, FOLDER_PATH+folderName+'/'+folderName.split('_')[-1]+'_'+fileName.split('img')[-1])
            print('rename '+fileName)
        os.system('mv {0}{1}/* {0}'.format(FOLDER_PATH, folderName))
'''
