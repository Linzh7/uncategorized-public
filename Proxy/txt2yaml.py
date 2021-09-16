f1 = open("1.txt", "r")
f2 = open("2.txt", "w")
ls = f1.readlines()
for ele in ls:
    if ele != "\n" and ele[0] != "#":
        ele = "- " + ele
        print(ele, file=f2, end="")
    else:
        print(ele, file=f2, end="")

f1.close()
f2.close()
