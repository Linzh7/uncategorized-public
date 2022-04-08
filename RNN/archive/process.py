from LinzhUtil import *
import pandas as pd
import threading
import time
from LinzhUtil import *


def excel2csv(path, excel):
    print("Processing on {}...".format(file))
    file_df = pd.read_excel("./dataset/{}{}".format(path, excel))
    file_df.to_csv("./dataset/output/{}".format(excel.split(".")[0]+".csv"))
    print("Task on {} is done.".format(file))


if __name__ == "__main__":
    folderList = getFolderList("./dataset/")
    for folder in folderList:
        if folder in ["__pycache__", "output"]:
            continue
        print("Processing in folder {}...".format(folder))
        fileList = getFileList("./dataset/"+folder)
        for file in fileList:
            if file.split(".")[-1] in ["xlsx", "xls"]:
                excel2csv(folder+"/", file)
                # threading.Thread(target=excel2csv, args=(folder+"/", file)).start()
                # time.sleep(5)
                break
        break
