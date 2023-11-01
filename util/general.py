import os
import time
import shutil
import hashlib
import torch


# Start
def md5_file(file_path):
    md5_obj = hashlib.md5()
    with open(file_path, 'rb') as file_obj:
        md5_obj.update(file_obj.read())
    file_md5_id = md5_obj.hexdigest()
    vid = file_md5_id[::4]  
    return vid, file_md5_id


def create_folder(path):
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        # print(path + " creates successfully.")
    # else:
    #     print(path + " already exists.")


def del_file(filepath):

    if os.path.exists(filepath):
        print(f"\t RM： {filepath}.")
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        else:
            os.remove(filepath)


def copy_file(src, dst):
    if os.path.exists(src):
        if os.path.exists(dst):
            del_file(dst)

        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)


def rename_dir_file(dir_path):
    files = sorted(os.listdir(dir_path))
    for i, file in enumerate(files):
        old_name = os.path.sep.join([dir_path, file])
        new_name = os.path.sep.join([dir_path, "{0:.8f}".format(time.perf_counter()) + "-" + file])
        os.rename(old_name, new_name)


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()