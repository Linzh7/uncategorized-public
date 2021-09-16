from operator import truediv
import os


def getFileList(filePath):
    for a, b, files in os.walk(filePath):
        return files


rootPath = 'E:\\Spider\\'
lable = 'horse'
sourcePath = rootPath + lable + '\\'
targetPath = rootPath + 'Done\\' + lable+'\\'

finalPath = targetPath

fileList = getFileList(finalPath)
for fileName in fileList:
    nameWithoutExtension = fileName.split('.')[0]
    namePath = finalPath+nameWithoutExtension+'.'
    ok = False
    if os.path.isfile(namePath+'xml'):
        if(os.path.isfile(namePath+'jpeg') or os.path.isfile(namePath+'jpg') or os.path.isfile(namePath+'png') or os.path.isfile(namePath+'gif')):
            ok = True
    if ok == False:
        os.system('DEL {}'.format(namePath+'xml'))
        os.system('DEL {}'.format(namePath+'jpg'))
        os.system('DEL {}'.format(namePath+'jpeg'))
        os.system('DEL {}'.format(namePath+'png'))
        os.system('DEL {}'.format(namePath+'gif'))
        print('del '+fileName)
    break
