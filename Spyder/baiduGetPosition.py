# encoding:utf-8
import requests
import base64
import threading
import os
import TOKEN
import time
import xml.etree.ElementTree as ET
from PIL import Image

rootPath = '/Users/Linzh/Local/GitHub/uncategorized/spyder/'
lable = 'pig'
sourcePath = rootPath + lable + '/'
targetPath = rootPath + 'Done/' + lable + '/'


def getFileList(filePath):
    for a, b, files in os.walk(filePath):
        return files


def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def getResponse(fileName, targetPath):
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"

    fileNamePath = sourcePath+fileName
    f = open(fileNamePath, 'rb')
    img = base64.b64encode(f.read())
    f.close()
    params = {"image": img, "with_face": 1}
    access_token = TOKEN.Baidu_Token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    try:
        response = requests.post(request_url, data=params, headers=headers)
        xyDict = eval(response.text)['result']
    except Exception:
        print('Error: '+fileName)
        os.system('mv {} {}'.format(sourcePath+fileName, sourcePath+'fail\\'+fileName))
        return
    print(fileName+' get info.')

    index = fileName.split(".")[0]
    root = ET.Element('annotation')
    tree = ET.ElementTree(root)

    ele = ET.SubElement(root, 'folder')
    ele.text = lable

    ele = ET.SubElement(root, 'filename')
    ele.text = fileName

    ele = ET.SubElement(root, 'path')
    ele.text = fileNamePath

    ele = ET.SubElement(root, 'source')
    child = ET.SubElement(ele, 'database')
    child.text = 'Unknown'

    ele = ET.SubElement(root, 'size')
    imgSize = Image.open(fileNamePath).size
    child = ET.SubElement(ele, 'width')
    child.text = str(imgSize[0])
    child = ET.SubElement(ele, 'height')
    child.text = str(imgSize[1])
    child = ET.SubElement(ele, 'depth')
    child.text = '3'
    root.append(ele)

    ele = ET.SubElement(root, 'segmented')
    ele.text = '0'

    ele = ET.SubElement(root, 'object')
    child = ET.SubElement(ele, 'name')
    child.text = lable
    child = ET.SubElement(ele, 'pose')
    child.text = 'Unspecified'
    child = ET.SubElement(ele, 'truncated')
    child.text = '1'
    child = ET.SubElement(ele, 'difficult')
    child.text = '0'
    bbox = ET.SubElement(ele, 'bndbox')
    child = ET.SubElement(bbox, 'xmin')
    child.text = str(xyDict['left'])
    child = ET.SubElement(bbox, 'ymin')
    child.text = str(xyDict['top'])
    child = ET.SubElement(bbox, 'xmax')
    child.text = str(xyDict['left'] + xyDict['width'])
    child = ET.SubElement(bbox, 'ymax')
    child.text = str(xyDict['top'] + xyDict['height'])

    __indent(root)
    tree.write(sourcePath+index+'.xml', encoding='utf-8', xml_declaration=False)

    print(fileName+" write into file.")
    # break
    xmlName = fileName.split('.')[0]+'.xml'
    os.system('mv {} {}'.format(sourcePath+fileName, targetPath+lable+fileName))
    os.system('mv {} {}'.format(sourcePath+xmlName, targetPath+lable+xmlName))
    print(fileName+" mv.")


def copyFileAuto(sourcePath, targetPath):
    fileList = getFileList(sourcePath)
    fileList.sort()
    for i in range(len(fileList)-1):
        tmp1 = fileList[i]
        tmp2 = fileList[i+1]
        if tmp1.split('.')[0] == tmp2.split('.')[0]:
            os.system('mv {} {}'.format(sourcePath+fileList[i], targetPath+lable+fileList[i]))
            os.system('mv {} {}'.format(sourcePath+fileList[i+1], targetPath+lable+fileList[i+1]))
            print(fileList[i]+' done.')


def createDir(targetPath):
    if not os.path.exists(targetPath):
        os.mkdir(targetPath)


createDir(targetPath)
copyFileAuto(sourcePath, targetPath)
fileList = getFileList(sourcePath)
if len(fileList) != 0:
    print(fileList)
    for imgFile in fileList:
        if imgFile.split('.')[-1] in ['jpg', 'jpge', 'png', 'bmp']:
            threading.Thread(target=getResponse, args=(imgFile, targetPath,)).start()
            time.sleep(0.09)
