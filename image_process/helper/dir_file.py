
import os
import stat

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        os.makedirs(path)
        os.chmod(path,stat.S_IREAD|stat.S_IWRITE)
        print ("---  new folder...  ---")
        print ("---  OK  ---")
    else:
        print ("---  There is this folder!  ---")
		
# file = "G:\\xxoo\\test"
# mkdir(file)             #调用函数


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
