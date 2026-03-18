import os
import pickle
from collections import defaultdict
from scipy.io import loadmat
from dassl.data.datasets.base_dataset import DatasetBase, Datum
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


class StanfordCars(DatasetBase):
    """Stanford Cars dataset for vehicle classification."""

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            total_train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")
            trainval = self.read_data("cars_train", trainval_file, meta_file)
            test = self.read_data("cars_test", test_file, meta_file)
            total_train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(total_train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        if num_shots >= 1:
            seed = cfg.SEED
            if cfg.TRAINER.NAME == "Baseline":
                preprocessed = os.path.join(self.baseline_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(total_train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                # with open(preprocessed, "wb") as file:
                #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        if cfg.DATASET.USERS > 0 and cfg.DATASET.SUBSAMPLE_CLASSES != 'all':
            federated_train_x = self.generate_federated_fewshot_dataset(train, num_shots=num_shots,
                                                                        num_users=cfg.DATASET.USERS,
                                                                        is_iid=cfg.DATASET.IID,
                                                                        repeat_rate=cfg.DATASET.REPEATRATE)
            if cfg.DATASET.USEALL:
                federated_test_x = defaultdict(list)
                for idx in range(cfg.DATASET.USERS):
                    federated_test_x[idx] = test
            else:
                federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                   num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                   repeat_rate=cfg.DATASET.REPEATRATE)
        elif cfg.DATASET.USERS > 0 and cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(total_train, num_shots=num_shots,
                                                                        num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            print("federated all dataset")
        elif cfg.DATASET.USERS > 0 and not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(total_train, num_shots=num_shots,num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            print("fewshot federated dataset")
        else:
            federated_train_x = None

        super().__init__(train_x=train, federated_train_x=federated_train_x, val=val, federated_test_x=federated_test_x, test=test)

    def read_data(self, image_dir: str, anno_file: str, meta_file: str):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items 