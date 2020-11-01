
import os
from importlib import import_module


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def getImportModuleName(configPath):
    if configPath.endswith(".py"):
        start = configPath.find("configs")
        endidx = configPath.find(".py")
        pakage = configPath[start:endidx]
        pakage = pakage.replace("/", ".")
    return pakage

class Config:
    def __init__(self, configfile):
        check_file_exist(configfile)
        pakage = getImportModuleName(configfile)
        self.config = import_module(pakage)
        # print(a)





if __name__ =="__main__":
    cfg = Config("/home/jinghuan/pythonProgram/implict3Dreconstruct/configs/BAE_AE.py")
    print(cfg.config.model)