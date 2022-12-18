from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import transforms3d as t3d

def compute_rel_pose(p1, p2):
    t1 = p1[:3]
    q1 = p1[3:]
    rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

    t2 = p2[:3]
    q2 = p2[3:]
    rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

    t_rel = t2 - t1
    rot_rel = np.dot(np.linalg.inv(rot1), rot2)
    q_rel = t3d.quaternions.mat2quat(rot_rel)
    return np.concatenate((t_rel, q_rel))


class KNNCameraPoseDataset(Dataset):
    """
        A class representing a dataset of netvlad encodings and their knn neighbors
    """

    def __init__(self, dataset_path, query_labels_file, db_labels_file, knn_file,
                 sample_neighbors=False, sample_size=20):
        super(KNNCameraPoseDataset, self).__init__()

        self.query_img_paths, self.query_poses, self.query_scenes, self.query_scenes_ids, self.query_netvlads_paths = read_labels_file(query_labels_file, dataset_path)
        self.query_to_pose = dict(zip(self.query_netvlads_paths, self.query_poses))

        self.db_img_paths, self.db_poses, self.db_scenes, self.db_scenes_ids, self.db_netvlads_paths = read_labels_file(db_labels_file, dataset_path)
        self.db_to_pose = dict(zip(self.db_netvlads_paths, self.db_poses))

        knns = {}
        lines = open(knn_file).readlines()
        for l in lines:
            neighbors = l.rstrip().split(",")
            q = join(dataset_path, neighbors[0].replace("_netvlad.npz", ".pmg").replace("_", "/")) #join(join(dataset_path, "netvlads"), neighbors[0])
            my_knns = []
            for nn in neighbors[1:]:
                #my_knns.append(join(join(dataset_path, "netvlads"), nn))
                my_knns.append(join(dataset_path, nn.replace("_netvlad.npz", ".pmg").replace("_", "/")))
            knns[q] = my_knns
        self.knns = knns
        self.sample = sample_neighbors
        self.sample_size = sample_size

    def __len__(self):
        return len(self.query_netvlads_paths)

    def __getitem__(self, idx):

        query_path = self.query_netvlads_paths[idx]
        knn_paths = self.knns[query_path]

        if self.sample:
            indices = np.random.choice(len(knn_paths), size=self.sample_size)
            knn_paths = np.array(knn_paths)[indices]
        else:
            knn_paths = knn_paths[:self.sample_size]

        query = np.load(query_path)["arr_0"]
        query_pose = self.query_to_pose[query_path]

        knn = np.zeros((self.sample_size, 4096))
        knn_poses = np.zeros((self.sample_size, 7))
        rel_poses = np.zeros((self.sample_size, 7))
        for i, nn_path in enumerate(knn_paths):
            knn[i, :] = np.load(nn_path)["arr_0"]
            knn_poses[i, :] = self.db_to_pose[nn_path]
            rel_poses[i, :] = compute_rel_pose(knn_poses[i, :], query_pose)

        return {"query":query, "query_pose":query_pose, "knn":knn, "knn_poses":knn_poses, "rel_poses": rel_poses}


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    netvlad_paths = [join(join(dataset_path, "netvlads"), path) for path in df['netvlad_path'].values]
    return imgs_paths, poses, scenes, scenes_ids, netvlad_paths