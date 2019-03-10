import os, glob

datasets_path = '/home/grupo08/M5/dataset'
problem_type = 'classificator' # Option: ['segmentator','classificator']
dataset_name = 'TT100K_trafficSigns'


#Load dataset paths and groundtruth information
train_images_path = os.path.join(datasets_path, problem_type, dataset_name, 'train')
valid_images_path = os.path.join(datasets_path, problem_type, dataset_name, 'valid')
test_images_path = os.path.join(datasets_path, problem_type, dataset_name, 'test')

train_images_txt = os.path.join(datasets_path, problem_type, dataset_name, dataset_name+'_train_images.txt')
valid_images_txt = os.path.join(datasets_path, problem_type, dataset_name, dataset_name+'_valid_images.txt')
test_images_txt = os.path.join(datasets_path, problem_type, dataset_name, dataset_name+'_test_images.txt')

train_gt_txt = os.path.join(datasets_path, problem_type, dataset_name, dataset_name+'_train_gt.txt')
valid_gt_txt = os.path.join(datasets_path, problem_type, dataset_name, dataset_name+'_valid_gt.txt')
test_gt_txt = os.path.join(datasets_path, problem_type, dataset_name, dataset_name+'_test_gt.txt')

def get_volume(dataset_path):
    labels_volume = {}
    for label in os.listdir(dataset_path):
        files = glob.glob(os.path.join(dataset_path, label, "*"))
        labels_volume[label] = len(files)
    return labels_volume


if __name__ == '__main__':
    print('TRAIN distribution: ' get_volume(train_images_path))
    print('VALID distribution: ' get_volume(valid_images_path))
    print('TEST distribution: ' get_volume(test_images_path))
