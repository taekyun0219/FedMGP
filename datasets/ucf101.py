import os
import pickle
import re
from collections import defaultdict
from dassl.data.datasets.base_dataset import DatasetBase, Datum
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


class UCF101(DatasetBase):
    """UCF101 dataset for action recognition."""

    dataset_dir = "ucf101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            total_train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            cname2lab = {}
            filepath = os.path.join(self.dataset_dir, "ucfTrainTestlist/classInd.txt")
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    label = int(label) - 1  # convert to 0-based index
                    cname2lab[classname] = label

            trainval = self.read_data(cname2lab, "ucfTrainTestlist/trainlist01.txt")
            test = self.read_data(cname2lab, "ucfTrainTestlist/testlist01.txt")
            total_train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(total_train, val, test, self.split_path, self.image_dir)

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
                federated_test_x = self.generate_federated_dataset(test, num_shots=-1,
                                                                   num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                   repeat_rate=cfg.DATASET.REPEATRATE)
        elif cfg.DATASET.USERS > 0 and cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(total_train, num_shots=num_shots,
                                                                        num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=-1,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            print("federated all dataset")
        elif cfg.DATASET.USERS > 0 and not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(total_train, num_shots=num_shots,num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE)
            federated_test_x = self.generate_federated_dataset(test, num_shots=-1,
                                                                num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID,
                                                                repeat_rate=cfg.DATASET.REPEATRATE)
            print("fewshot federated dataset")
        else:
            federated_train_x = None

        super().__init__(train_x=train, federated_train_x=federated_train_x, val=val, federated_test_x=federated_test_x, test=test)

    def read_data(self, cname2lab: dict, text_file: str):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(impath=impath, label=label, classname=renamed_action)
                items.append(item)

        return items 
