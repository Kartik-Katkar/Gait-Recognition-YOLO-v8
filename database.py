import gc
import os
import sqlite3
import argparse
from datetime import datetime
from config import conf
from util.general import create_folder, del_file

DATABASE = 'db/user_data.db'


def get_db():

    try:
        db = sqlite3.connect(DATABASE)
        cur = db.cursor()
        return db, cur
    except Exception as error:
        print(error)


def close_db(db, cur):

    db.commit()
    cur.close()
    db.close()
    gc.collect()
    return True


def person_register(person_name, vid, vmd5, vname):
    if person_name == "":
        pid = "probe"
        create_person_folder(pid)
        return True, pid


    if pid := get_pid_from_name(person_name):
        tag = False
        print(f"\t {person_name} exists, {pid=}.")
    else:  
        tag = True
        pid = create_person_data(person_name)
        create_person_folder(pid)


    create_video_data(vid, pid, vmd5, vname)
    return tag, str(pid)


def create_person_data(pname, gender=None, age=None, email=None, phone=None, address=None, other=None, ptag=None,
                       timetag=None):
    print(f"\t Create person: {pname=}.")


    db, cur = get_db()

    timetag = timetag if timetag else datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
    cur.execute(
        "INSERT INTO person (pname, gender, age, email, phone, address, other, ptag, timetag) VALUES(?,?,?,?,?,?,?,?,?)",
        [pname, gender, age, email, phone, address, other, ptag, timetag])
    pid = cur.lastrowid


    close_db(db, cur)

    return pid


def create_person_folder(pid):
    print(f"\t Create person folder: {pid=}.")
    pid = str(pid)
    person_folder = os.path.sep.join([conf['UPLOAD_FOLDER'], pid])
    person_video_folder = os.path.sep.join([person_folder, 'video'])
    person_image_folder = os.path.sep.join([person_folder, 'image'])
    person_silhouette_folder = os.path.sep.join([person_folder, 'silhouette'])
    person_pkl_folder = os.path.sep.join([person_folder, 'pkl'])
    create_folder(person_video_folder)
    create_folder(person_image_folder)
    create_folder(person_silhouette_folder)
    create_folder(person_pkl_folder)


def update_person_data(pid, pname, gender, age, email, phone, address, other, ptag, timetag=None):
    print(f"\t Update person: {pid=}.")

    db, cur = get_db()

    timetag = timetag if timetag else datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 生成时间戳
    cur.execute(
        "UPDATE person SET pname = ?, gender = ?, age = ?,  email = ?, phone = ?, address = ?, other = ?, ptag = ? timetag = ? where pid = ?",
        [pname, gender, age, email, phone, address, other, ptag, timetag, pid])


    close_db(db, cur)

    return True


def get_pid_from_name(person_name):

    db, cur = get_db()
    cur.execute("SELECT pid FROM person WHERE pname = ?", [person_name])
    result = cur.fetchone()
    pid = result[0] if result else None
    close_db(db, cur)
    return pid


def delete_person(pid="0", pname=None):
    if not pid.isdigit():
        print(f"\t Invalid pid.")
        return False

    pid = get_pid_from_name(pname) if pname else int(pid)
    print(f"\t Delete person: {pid=}.")


    db, cur = get_db()
    cur.execute("DELETE FROM video where pid = ?", [pid])
    cur.execute("DELETE FROM person where pid = ?", [pid])
    close_db(db, cur)


    pid = str(pid)
    person_folder = os.path.sep.join([conf["UPLOAD_FOLDER"], pid])
    person_datasets_folder = os.path.sep.join([conf["DATASETS_FOLDER"], pid])
    if os.path.exists(person_folder):
        del_file(person_folder)
    if os.path.exists(person_datasets_folder):
        del_file(person_datasets_folder)

    return True


def create_video_data(vid, pid, vmd5, vname, vdesc=None, vpath=None, vtag=None, timetag=None):
    print(f"\t Create video data: {vid=}.")


    db, cur = get_db()

    timetag = timetag if timetag else datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 生成时间戳
    cur.execute(
        "INSERT INTO video (vid, pid, vmd5, vname, vdesc, vpath, vtag, timetag) VALUES(?,?,?,?,?,?,?,?)",
        [vid, pid, vmd5, vname, vdesc, vpath, vtag, timetag])


    close_db(db, cur)

    return True


def update_video_data(vid, pid, vmd5, vname, vdesc, vpath, vtag, timetag=None):
    print(f"\t Update video data: {vid=}.")

    db, cur = get_db()

    timetag = timetag if timetag else datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 生成时间戳
    cur.execute(
        "UPDATE video SET pid = ?, vmd5 = ?, vname = ?, vdesc = ?, vpath = ?, vtag = ?, timetag = ? where vid = ?",
        [pid, vmd5, vname, vdesc, vpath, vtag, timetag, vid])


    close_db(db, cur)

    return True


def get_pid_vname_from_vid(vid):

    db, cur = get_db()
    cur.execute("SELECT pid, vname FROM video WHERE vid = ?", [vid])
    result = cur.fetchone()
    pid, vname = result if result else (None, None)
    close_db(db, cur)
    return pid, vname


def get_pname_from_vid(vid):
    db, cur = get_db()
    cur.execute("SELECT pname from person where pid IN(SELECT pid from video WHERE vid = ?)", [vid])
    result = cur.fetchone()
    pname = result[0] if result else "None"
    close_db(db, cur)
    return pname


def md5_exists(vmd5):

    db, cur = get_db()
    cur.execute("SELECT vid FROM video WHERE vmd5 = ?", [vmd5])
    result = cur.fetchone()
    vid = result[0] if result else None
    close_db(db, cur)
    return vid


def delete_video(vid):
    print(f"\t Delete video data: {vid=}.")


    db, cur = get_db()

    pid, vname = get_pid_vname_from_vid(vid)
    print(f"\t {pid=}, {vname=}")

    if pid:

        delete_video_file(pid, vid, vname)

        cur.execute("DELETE FROM video where vid = ?", [vid])

    close_db(db, cur)
    return True


def delete_video_file(pid, vid, vname=None):
    print(f"\t Delete video file: {vid=}.")
    video_folder = os.path.sep.join([conf["UPLOAD_FOLDER"], pid, "video"])
    if not vname: 
        for name in os.listdir(video_folder):
            if name.split(".")[0] == vid:
                vname = name
                break
    video_file = os.path.sep.join([video_folder, str(vname)])
    video_datasets_folder = os.path.sep.join([conf["DATASETS_FOLDER"], pid, vid])
    image_file = os.path.sep.join([conf["UPLOAD_FOLDER"], pid, "image", vid])
    pkl_file = os.path.sep.join([conf["UPLOAD_FOLDER"], pid, "pkl", vid])
    silhouette_file = os.path.sep.join([conf["UPLOAD_FOLDER"], pid, "silhouette", vid])
    cut_image_file = os.path.sep.join([conf["UPLOAD_FOLDER"], pid, "cut_img", vid])

    del_file(video_file)
    del_file(video_datasets_folder)
    del_file(image_file)
    del_file(pkl_file)
    del_file(silhouette_file)
    del_file(cut_image_file)


parser = argparse.ArgumentParser(description='Database')
parser.add_argument('--delete_video', default='', type=str, help='delete video by vid')
parser.add_argument('--delete_person', default='', type=str, help='delete person by pid')
parser.add_argument('--delete_person_by_name', default='', type=str, help='delete person by name')
parser.add_argument('--delete_probe_video', default='', type=str, help='delete probe video by vid')
opt = parser.parse_args()

# opt.delete_person = ""
# opt.delete_person_by_name = ""
# opt.delete_video = "90302d25"
# opt.delete_probe_video = "90302d25"

if __name__ == '__main__':
    if opt.delete_video:
        delete_video(opt.delete_video)
    if opt.delete_person:
        delete_person(opt.delete_person)
    if opt.delete_person_by_name:
        delete_person(pname=opt.delete_person_by_name)
    if opt.delete_probe_video:
        delete_video_file('probe', opt.delete_probe_video)
