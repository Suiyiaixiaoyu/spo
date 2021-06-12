import argparse

def getparse():
    parse = argparse.ArgumentParser()
    parse.add_argument("--GPUNUM",default = 0,type=int)
    parse.add_argument("--vocab_path",default= 'bert/vocab.txt',type=str)
    parse.add_argument("--train_path",default='data/train.json',type=str)
    parse.add_argument("--dev_path",default='data/dev.json',type=str)
    parse.add_argument("--schemas_path",default='data/schemas.json',type=str)
    parse.add_argument("--bert_path",default='bert',type=str)
    parse.add_argument("--lr",default = 5e-5,help='learning_rate' )
    parse.add_argument("--hidden",default = 1024,help='hidden_size')



    return parse

