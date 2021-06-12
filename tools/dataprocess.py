import json


def loadvocab(path):
    vocab = {}
    with open(path,encoding='utf-8') as file:
        for l in file.readlines():
            vocab[len(vocab)] = l.strip()
    return vocab

def loaddata(path):
    with open(path,encoding='utf-8') as file:
        jsonlist = json.load(file)

    return jsonlist

def loadschemas(path):
    with open(path,encoding='utf-8') as file:
        jsonlist = json.load(file)
        idtopredicate = jsonlist[0]
        predicatetoid = jsonlist[1]
    return idtopredicate,predicatetoid

