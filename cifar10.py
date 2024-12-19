import argparse
import random
import sys
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
from collections import Counter

sys.path.append("../../")
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.logger import Logger
from fedlab.utils.serialization import SerializationTool
from fedlab.models.mlp import MLP
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.algorithm.basic_server import SyncServerHandler


import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def print_client_data_counts(dataset):
    client_data_counts = {client_id: len(indices) for client_id, indices in dataset.data_indices.items()}
    print("Each client's data sample count:", client_data_counts)
    for client_id, indices in dataset.data_indices.items():
        label_counts = Counter([dataset.dataset[i][1] for i in indices])
        print(f"Client {client_id} label distribution: {dict(label_counts)}")

def plot_client_distribution(dataset):
    client_data_counts = {client_id: len(indices) for client_id, indices in dataset.data_indices.items()}
    # 获取每个客户端的类别分布
    all_label_counts = []
    all_clients = sorted(dataset.data_indices.keys())
    for client_id in all_clients:
        label_counts = Counter([dataset.dataset[i][1] for i in dataset.data_indices[client_id]])
        all_label_counts.append(label_counts)

    all_labels = sorted(list(set([label for lc in all_label_counts for label in lc.keys()])))
    num_clients = len(all_clients)
    num_classes = len(all_labels)
    data_matrix = []
    for lc in all_label_counts:
        row = []
        for c in all_labels:
            row.append(lc.get(c, 0))
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    bottoms = np.zeros(num_clients)
    colors = plt.cm.get_cmap('tab10', num_classes)
    for i, c in enumerate(all_labels):
        ax[0].barh(all_clients, data_matrix[:, i], left=bottoms, label=f'class{c}', color=colors(i))
        bottoms += data_matrix[:, i]

    ax[0].set_xlabel("Number of Samples")
    ax[0].set_ylabel("Client")
    ax[0].set_title("Class Distribution per Client")
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    counts = list(client_data_counts.values())
    ax[1].hist(counts, bins=20, edgecolor='black', alpha=0.7)
    ax[1].set_xlabel("num_samples")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Distribution of Number of Samples per Client")

    plt.tight_layout()
    plt.savefig("client_distribution.png", dpi=300)
    # plt.show() # 可注释掉


# 自定义数据集类
class CustomDataset:
    def __init__(self, dataset, num_clients, partition_type="iid", alpha=0.5):
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.alpha = alpha
        self.data_indices = self.partition_data()

    def partition_data(self):
        num_samples = len(self.dataset) // self.num_clients
        if self.partition_type == "iid":
            return {i: list(range(i * num_samples, (i + 1) * num_samples)) for i in range(self.num_clients)}
        elif self.partition_type == "dirichlet":
            return self.dirichlet_partition()
        elif self.partition_type == "lognormal":
            return self.lognormal_partition()
        elif self.partition_type == "random":
            return self.random_partition()
        else:
            raise ValueError("Invalid partition type.")

    def dirichlet_partition(self):
        """Dirichlet分布生成非IID数据"""
        labels = [label for _, label in self.dataset]
        indices = [[] for _ in range(self.num_clients)]
        for cls in set(labels):
            cls_indices = [i for i, label in enumerate(labels) if label == cls]
            proportions = torch.distributions.Dirichlet(torch.tensor([self.alpha] * self.num_clients)).sample()
            proportions = (proportions / proportions.sum()).tolist()
            split_indices = [int(prop * len(cls_indices)) for prop in proportions]
            current_idx = 0
            for i, num in enumerate(split_indices):
                indices[i].extend(cls_indices[current_idx:current_idx + num])
                current_idx += num
        return {i: indices[i] for i in range(self.num_clients)}

    def lognormal_partition(self):
        """Lognormal分布生成非IID数据"""
        labels = [label for _, label in self.dataset]
        indices = [[] for _ in range(self.num_clients)]
        for cls in set(labels):
            cls_indices = [i for i, label in enumerate(labels) if label == cls]
            proportions = torch.distributions.LogNormal(0, self.alpha).sample([self.num_clients])
            proportions = (proportions / proportions.sum()).tolist()
            split_indices = [int(prop * len(cls_indices)) for prop in proportions]
            current_idx = 0
            for i, num in enumerate(split_indices):
                indices[i].extend(cls_indices[current_idx:current_idx + num])
                current_idx += num
        return {i: indices[i] for i in range(self.num_clients)}

    def random_partition(self):
        """随机分片数据"""
        indices = torch.randperm(len(self.dataset)).tolist()
        shard_size = len(self.dataset) // (self.num_clients * 2)
        shards = [indices[i:i + shard_size] for i in range(0, len(indices), shard_size)]
        client_indices = {i: [] for i in range(self.num_clients)}
        for client in range(self.num_clients):
            shard_ids = random.sample(range(len(shards)), 2)
            client_indices[client].extend(shards[shard_ids[0]])
            client_indices[client].extend(shards[shard_ids[1]])
        return client_indices

    def get_dataloader(self, client_id, batch_size):
        indices = self.data_indices[client_id]
        client_dataset = Subset(self.dataset, indices)
        return DataLoader(client_dataset, batch_size=batch_size, shuffle=True)


# 配置命令行参数
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=3)
parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--partition_type", type=str, default="iid", choices=["iid", "dirichlet", "lognormal", "random"])
parser.add_argument("--alpha", type=float, default=0.5)  # Dirichlet/Lognormal 参数
args = parser.parse_args()

# # 配置模型和日志
# model = MLP(784, 10)
# Update model instantiation
model = MLP(3072, 10)  # 3072 input size for CIFAR-10, 10 classes

logger = Logger(log_name="standalone")

# # 初始化数据集
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

from torchvision.transforms import Compose, ToTensor, Resize, Grayscale

transform = Compose([
    Resize((28, 28)),  # Resize CIFAR-10 images to 28x28
    Grayscale(num_output_channels=1),  # Convert RGB to grayscale
    ToTensor()
])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
])

# Download CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


# 创建自定义数据集以分割数据
train_data = CustomDataset(train_dataset, num_clients=args.total_client, partition_type=args.partition_type, alpha=args.alpha)

# 初始化服务端和客户端处理程序
handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, logger=logger)
trainer = SGDSerialClientTrainer(model=model, num_clients=args.total_client, cuda=False)

# 设置训练器的自定义数据集
trainer.setup_dataset(train_data)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# 打印每个客户端的数据样本数量和类分布
def print_client_data_counts(dataset):
    client_data_counts = {client_id: len(indices) for client_id, indices in dataset.data_indices.items()}
    print("Each client's data sample count:", client_data_counts)
    for client_id, indices in dataset.data_indices.items():
        label_counts = Counter([dataset.dataset[i][1] for i in indices])
        print(f"Client {client_id} label distribution: {dict(label_counts)}")

print_client_data_counts(trainer.dataset)

# 自定义Pipeline
class CustomPipeline(StandalonePipeline):
    def main(self):
        for round in range(self.handler.global_round):
            print(f"\nStarting communication round {round + 1}/{self.handler.global_round}")
            sampled_clients = self.handler.sample_clients()
            print(f"Sampled clients: {sampled_clients}")

            self.trainer.local_process(self.handler.downlink_package, sampled_clients)
            uploads = self.trainer.uplink_package

            for pack in uploads:
                self.handler.load(pack)

            print("Clients have finished local training and sent updates back to the server.")
            self.handler.global_update(uploads)
            print("Server has aggregated the updates.")
            self.evaluate()

    def evaluate(self):
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        self.trainer.model.eval()
        total_loss, total_correct = 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                output = self.trainer.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()

        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = total_correct / len(test_loader.dataset) * 100
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


# Run the pipeline as before
custom_pipeline = CustomPipeline(handler=handler, trainer=trainer)
custom_pipeline.main()
# 调用
print_client_data_counts(trainer.dataset)
plot_client_distribution(trainer.dataset)


