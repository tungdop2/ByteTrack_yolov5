import json
import os
from tqdm import tqdm
import argparse

def mix_data(list_folder):
    os.mkdir('mix_det')
    os.mkdir('mix_det/annotations')

    img_list = list()
    ann_list = list()
    video_list = list()
    category_list = list()
    for i, folder in enumerate(list_folder):
        os.symlink('../' + folder, 'mix_det/' + folder + '_train/')
        if i == 0:
            data_json = json.load(open('datasets/' + folder + '/annotations/train.json','r'))
            print('img')
            for img in tqdm(data_json['images']):
                img['file_name'] = folder + '_train/' + img['file_name']
                img_list.append(img)

            print('ann')
            for ann in tqdm(data_json['annotations']):
                ann_list.append(ann)

            video_list = data_json['videos']
            category_list = data_json['categories']
            print('Add ' + folder + ' data successfully!')
        else:
            max_img = 10000 * i
            max_ann = 2000000 * i
            max_video = 10 * i
            data_json = json.load(open('datasets/' + folder + '/annotations/train.json','r'))

            img_id_count = 0
            print('img')
            for img in tqdm(data_json['images']):
                img_id_count += 1
                img['file_name'] = folder + '_train/' + img['file_name']
                img['frame_id'] = img_id_count
                img['prev_image_id'] = img['id'] + max_img
                img['next_image_id'] = img['id'] + max_img
                img['id'] = img['id'] + max_img
                img['video_id'] = max_video
                img_list.append(img)

            print('ann')
            for ann in tqdm(data_json['annotations']):
                ann['id'] = ann['id'] + max_ann
                ann['image_id'] = ann['image_id'] + max_img
                ann_list.append(ann)

            video_list.append({
                'id': max_video,
                'file_name': folder + '_train'
            })
            print('Add ' + folder + ' data successfully!')

    print('mix_data')
    print('img_list:', len(img_list))
    print('ann_list:', len(ann_list))
    print('video_list:', len(video_list))
    print('category_list:', len(category_list))
    mix_json = dict()
    mix_json['images'] = img_list
    mix_json['annotations'] = ann_list
    mix_json['videos'] = video_list
    mix_json['categories'] = category_list
    json.dump(mix_data, open('datasets/mix_det/annotations/train.json','w'))
    print('Mix_data successfully!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_folder', nargs='+', type=str, help='List of folder to mix data')
    args = parser.parse_args()
    mix_data(args.list_folder)
