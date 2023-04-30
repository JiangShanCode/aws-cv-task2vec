from task2vec import Task2Vec
from models import get_model
import datasets
import task_similarity

dataset_names = ('mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
dataset_list = [datasets.__dict__[name]('./data')[0] for name in dataset_names] 
# print(dataset_list)
embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"Embedding {name}")
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets)+1)).cuda()
    # embeddings.append( Task2Vec(probe_network,method='variational').embed(dataset) )
    print(Task2Vec(probe_network,method='montecarlo').embed(dataset).hessian)
# task_similarity.plot_distance_matrix(embeddings, dataset_names)

# from task2vec import Task2Vec
# from models import get_model
# import datasets

# config = {}

# dataset = datasets.__dict__["cifar10"]('./data')[0]
# probe_network = get_model('resnet34', pretrained=True, num_classes=10)
# embedding =  Task2Vec(probe_network).embed(dataset)
# print(embedding)