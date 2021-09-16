import os


def getFileList(filePath):
    for a, b, files in os.walk(filePath):
        return files


rootPath = 'E:\\Spider\\'
lable = 'cow'
sourcePath = rootPath + lable + '\\'
targetPath = rootPath + 'Done\\' + lable+'\\'
ok = True

fileList = getFileList(sourcePath)
for i in range(len(fileList)-1):
    tmp1 = fileList[i]
    tmp2 = fileList[i+1]
    if tmp1.split('.')[0] == tmp2.split('.')[0]:
        count = 0
        os.system('move {} {}'.format(sourcePath+fileList[i], targetPath+lable+fileList[i]))
        os.system('move {} {}'.format(sourcePath+fileList[i+1], targetPath+lable+fileList[i+1]))
        print(fileList[i]+' done.')
