from PIL import Image
import xml.etree.ElementTree as ET


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


def BaiduWriteIntoXML(fileName, path, response):
    try:
        ls = response['vehicle_info']

        root = ET.Element('annotation')
        tree = ET.ElementTree(root)

        child = ET.SubElement(root, 'folder')
        child.text = 'baidu'

        child = ET.SubElement(root, 'filename')
        child.text = fileName

        child = ET.SubElement(root, 'path')
        child.text = './'

        child = ET.SubElement(root, 'source')
        subchild = ET.SubElement(child, 'database')
        subchild.text = 'Unknown'

        child = ET.SubElement(root, 'size')
        subchild = ET.SubElement(child, 'width')
        img = Image.open(path+fileName)
        subchild.text = str(img.size[0])
        subchild = ET.SubElement(child, 'height')
        subchild.text = str(img.size[1])
        subchild = ET.SubElement(child, 'depth')
        subchild.text = '3'

        child = ET.SubElement(root, 'segmented')
        child.text = '0'

        for i in range(len(ls)):
            if ls[i]['probability'] < 0.1:
                continue
            if ls[i]['type'] in ['motorbike', 'tricycle']:
                continue
            child = ET.SubElement(root, 'object')
            subchild = ET.SubElement(child, 'name')
            subchild.text = ls[i]['type']
            subchild = ET.SubElement(child, 'pose')
            subchild.text = 'Unspecified'
            subchild = ET.SubElement(child, 'truncated')
            subchild.text = '1'
            subchild = ET.SubElement(child, 'difficult')
            subchild.text = '0'
            subchild = ET.SubElement(child, 'bndbox')
            xy = ET.SubElement(subchild, 'xmin')
            xy.text = str(ls[i]['location']['left'])
            xy = ET.SubElement(subchild, 'ymin')
            xy.text = str(ls[i]['location']['top'])
            xy = ET.SubElement(subchild, 'xmax')
            xy.text = str(ls[i]['location']['left']+ls[i]['location']['width'])
            xy = ET.SubElement(subchild, 'ymax')
            xy.text = str(ls[i]['location']['top']+ls[i]['location']['height'])

        __indent(root)
        tree.write(path+fileName.split('.')[0]+'.xml', encoding='utf-8', xml_declaration=False)
        print(fileName+' got info.')
    except:
        print('Error: '+response['error_msg'])


def YSwriteIntoXML(fileName, path, response):
    ls = response['data']

    root = ET.Element('annotation')
    tree = ET.ElementTree(root)

    child = ET.SubElement(root, 'folder')
    child.text = 'YS'

    child = ET.SubElement(root, 'filename')
    child.text = fileName

    child = ET.SubElement(root, 'path')
    child.text = './'

    child = ET.SubElement(root, 'source')
    subchild = ET.SubElement(child, 'database')
    subchild.text = 'Unknown'

    child = ET.SubElement(root, 'size')
    subchild = ET.SubElement(child, 'width')
    img = Image.open(path+fileName)
    subchild.text = str(img.size[0])
    subchild = ET.SubElement(child, 'height')
    subchild.text = str(img.size[1])
    subchild = ET.SubElement(child, 'depth')
    subchild.text = '3'

    child = ET.SubElement(root, 'segmented')
    child.text = '0'

    for i in range(len(ls)):
        if ls[i]['vehicleType']['val'] == 'threeWheelVehicle':
            continue
        child = ET.SubElement(root, 'object')
        subchild = ET.SubElement(child, 'name')
        if ls[i]['vehicleType']['val'] in ['van', 'unknown', 'vehicle', 'SUV/MPV']:
            text = 'car'
        else:
            text = ls[i]['vehicleType']['val']
        subchild.text = text
        subchild = ET.SubElement(child, 'pose')
        subchild.text = 'Unspecified'
        subchild = ET.SubElement(child, 'truncated')
        subchild.text = '1'
        subchild = ET.SubElement(child, 'difficult')
        subchild.text = '0'
        subchild = ET.SubElement(child, 'bndbox')
        xy = ET.SubElement(subchild, 'xmin')
        xy.text = str(ls[i]['rect']['x'])
        xy = ET.SubElement(subchild, 'ymin')
        xy.text = str(ls[i]['rect']['y'])
        xy = ET.SubElement(subchild, 'xmax')
        xy.text = str(ls[i]['rect']['x']+ls[i]['rect']['width'])
        xy = ET.SubElement(subchild, 'ymax')
        xy.text = str(ls[i]['rect']['y']+ls[i]['rect']['height'])

    __indent(root)
    tree.write(path+fileName.split('.')[0]+'.xml', encoding='utf-8', xml_declaration=False)
    print(fileName+' done')
