# encoding:utf-8
import requests
import base64
import TOKEN
import os
import xml.etree.ElementTree as ET
from PIL import Image

rootPath = 'E:\\Spider\\'
lable = 'horse'
Path = rootPath+lable+'\\'
ok = True


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


fileList = getFileList(Path)

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"

failList = []
error = 0

for fileName in fileList:
    if len(fileList) < 2:
        ok = False
        break
    if error > 1:
        break
    if fileName.split(".")[-1] == "xml":
        continue
    fileNamePath = Path+fileName
    f = open(fileNamePath, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img, "with_face": 1}
    access_token = TOKEN.Baidu_Token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    try:
        response = requests.post(request_url, data=params, headers=headers)
    except Exception as e:
        print(fileName+" "+str(e))
        failList.append(fileName)
        error += 1
        continue
    if response:
        print(fileName+" get info.")
        try:
            xyDict = eval(response.text)['result']
        except KeyError:
            print(fileName+" cannot get any result.")
            failList.append(fileName)
            error += 1
            continue
        except Exception as e:
            print(fileName+" "+str(e))
            failList.append(fileName)
            error += 1
            continue
    else:
        print(fileName+" no response.")

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
    tree.write(Path+index+'.xml', encoding='utf-8', xml_declaration=False)
    print(fileName+" write into file.")
    # break


print("Fail List:")
print(failList)
for ele in failList:
    os.system('move {} {}'.format(Path+ele, Path+'fail\\'+ele))


if ok:
    os.system('python copyFileAutoSlow.py')
