import os.path as osp
import hashlib
import os

md5sum = hashlib.md5

from utils import file_processing


def create_md5sum(image_dir, totle,md5sum_file):
    with open(md5sum_file, 'w') as f:
        for line in totle:
            image_name = line[0]
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                print("no path:{}".format(image_path))
            img = open(image_path, 'rb').read()
            md5 = md5sum(img).hexdigest()
            f.write(image_name + ' ' + str(md5) + '\n')
            print(image_name + ' ' + str(md5))


if __name__ == "__main__":
    filename = "/media/dm/dm/project/dataset/COCO/HumanPose/teacher_2D_pose_estimator/list/val.txt"
    image_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/teacher_2D_pose_estimator/list/val"
    md5sum_file="val_md5sum.txt"
    data = file_processing.read_data(filename)
    create_md5sum(image_dir, data,md5sum_file)
