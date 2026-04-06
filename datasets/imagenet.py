import os
import pickle
from collections import defaultdict, OrderedDict
from dassl.data.datasets.base_dataset import DatasetBase, Datum
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


class ImageNet(DatasetBase):
    """ImageNet dataset for large-scale image classification."""

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                total_train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            total_train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": total_train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ImageNet doesn't have a separate validation set in the original implementation
        val = test

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
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(total_train, num_shots=num_shots)
                data = {"train": train}
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

    @staticmethod
    def read_classnames(text_file: str) -> OrderedDict:
        """Read classnames from file. Returns folder-to-classname mapping."""
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames: OrderedDict, split_dir: str):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items 
