import os


def getFileList(filePath):
    for a, b, files in os.walk(filePath):
        return files


rootPath = 'E:\\Spider\\'
lable = 'horse'
sourcePath = rootPath + lable + '\\'
targetPath = rootPath + 'Done\\' + lable+'\\'
ok = True

fileList = getFileList(sourcePath)
length = len(fileList)-1
for i in range(length):
    tmp1 = fileList[i]
    tmp2 = fileList[i+1]
    if tmp1.split('.')[0] == tmp2.split('.')[0]:
        count = 0
        os.system('move {} {}'.format(sourcePath+fileList[i], targetPath+lable+fileList[i]))
        os.system('move {} {}'.format(sourcePath+fileList[i+1], targetPath+lable+fileList[i+1]))
        print(fileList[i]+' done.')
os.system('python baiduGetPositionSlow.py')
