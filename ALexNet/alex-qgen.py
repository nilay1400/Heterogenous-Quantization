import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import random
from deap import base, creator, tools, algorithms
import numpy as np


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(torch.__version__)
# print(DEVICE)


train_csv = pd.read_csv('./kaggle/input/fashionmnist/fashion-mnist_train.csv')
test_csv = pd.read_csv('./kaggle/input/fashionmnist/fashion-mnist_test.csv')

inputSize = 8000
train_csv=train_csv[:inputSize]
# len(train_csv)


class FashionDataset(Dataset):
    def __init__(self, data, transform=None):        
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        
        label, image = [], []
        
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        
        if self.transform is not None:
            # transfrom the numpy array to PIL image before the transform function
            pil_image = Image.fromarray(np.uint8(image)) 
            image = self.transform(pil_image)
            
        return image, label


AlexTransform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_loader = DataLoader(
    FashionDataset(train_csv, transform=AlexTransform), 
    batch_size=100, shuffle=False)

test_loader = DataLoader(
    FashionDataset(train_csv, transform=AlexTransform), 
    batch_size=100, shuffle=False)


class fasion_mnist_alexnet(nn.Module):  
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out



def testt(model):
    # model.eval()
    # test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += criterion(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(test_loader.dataset)  # loss之和除以data数量 -> mean
        # accuracy_val.append(100. * correct / len(test_loader.dataset))
        # print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            # test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        # print(test_loss)
        # print(correct)
        # print(accuracy_val)
        acc = 100. * correct / len(test_loader.dataset)
        return(acc)
    

model = torch.load('qalex-0-7.pth')    


import timeit

model = model.to('cuda')
    # model.eval()
start_time = timeit.default_timer()
dataloader =test_loader
model.eval()
acc = testt(model)
    # print(acc)ts.writerow(csv_output)
print('Total Time:', timeit.default_timer() - start_time)
print('Accuracy: %.4f %%' % (acc))


model = model.to('cuda')
result = testt(model)
print(result)

######....................parameter number............................######
layer_names = [name for name in model.state_dict().keys() if 'weight._data' in name]
print(layer_names)
num_weights_per_layer = {name: model.state_dict()[name].numel() for name in layer_names}
n = 0
for _ in layer_names:
  n = n + num_weights_per_layer[_]
print("Parameters Number:", n)

def get_nested_attr(obj, attr):
    try:
        # Split the string by '.' to get individual attributes and indices
        parts = attr.split('.')
        for part in parts:
            # Check if part is indexed (e.g., 'features[0]')
            if '[' in part and ']' in part:
                # Split by '[' and extract the index
                part, idx = part.split('[')
                idx = int(idx[:-1])  # Convert '0]' to 0
                obj = getattr(obj, part)[idx]
            else:
                obj = getattr(obj, part)
        return obj
    except AttributeError as e:
        print(f"Error: {e}")
        return None

model = model.to('cuda')
# Define the evaluation function
def evaluate(individual):
    # print("Evaluating individual:", individual)
    model_copy = torch.load('qalex-0-7.pth')
    model_copy = model_copy.to('cuda')

###################################################################

    state_dict = model_copy.state_dict()
    for layer_name, weight_idx in individual:
        weight = get_nested_attr(model, layer_name).view(-1).to('cuda')


        
        bit_position = random.randint(0, 2)
        #print(f"bit position:{iibit}")

        
        flipped_value = weight[weight_idx] ^ (1 << bit_position)
        weight[weight_idx] = flipped_value


    # Load the perturbed weights back into the model
    model_copy.load_state_dict(state_dict)

    # Evaluate the perturbed model on a validation set
    model_copy.eval()
    result = testt(model_copy)
    Accuracy = result
    loss = acc - Accuracy


    # Return the loss as fitness (higher loss indicates more critical weight)
    return loss,

# Create the fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize the function
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def custom_mutate(individual, indpb):
    # print("Before mutation:", individual)
    for i in range(len(individual)):
        if random.random() < indpb:
            layer, index = individual[i]
            new_index = random.randint(0, num_weights_per_layer[layer] - 1)
            individual[i] = (layer, new_index)
    # print("After mutation:", individual)
    return individual,

def custom_crossover(ind1, ind2):
    # print("Before crossover:", ind1, ind2)
    tools.cxTwoPoint(ind1, ind2)
    # print("After crossover:", ind1, ind2)
    return ind1, ind2


# Attribute generator: (layer, index) pair
layer_names = [name for name in model.state_dict().keys() if 'weight._data' in name]
num_weights_per_layer = {name: model.state_dict()[name].numel() for name in layer_names}
def random_weight():
    layer = random.choice(layer_names)
    index = random.randint(0, num_weights_per_layer[layer] - 1)
    # print(layer,index)
    return (layer, index)

alpha = 0.00001
ind_size = int(alpha * n)
print("Individual Size =", ind_size)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, random_weight, n=ind_size)  # Each individual perturbs 5 weights
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("mate", custom_crossover)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


from deap import tools, base, creator, algorithms

def eaSimpleWithDebugging(population, toolbox, cxpb, mutpb, ngen, stats=None,
                          halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        print(f"Generation {gen}")

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                # print(f"Before mate: {child1}, {child2}")
                toolbox.mate(child1, child2)
                # print(f"After mate: {child1}, {child2}")
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                # print(f"Before mutate: {mutant}")
                toolbox.mutate(mutant)
                # print(f"After mutate: {mutant}")
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
    start_time2 = timeit.default_timer()
    random.seed(42)

    # Create an initial population of 100 individuals
    population = toolbox.population(n=100)

    # Define statistics to keep track of the progress
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Extract the first element of the fitness tuple
    stats.register("avg", np.mean)
    stats.register("min", min)
    stats.register("max", max)

    # Hall of Fame to keep the best individual
    hof = tools.HallOfFame(1)

    # Run the genetic algorithm
    population, logbook = eaSimpleWithDebugging(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                                stats=stats, halloffame=hof, verbose=True)

    # Print the best individual
    print("Best individual is: ", hof[0])
    print("Fitness: ", hof[0].fitness.values[0])
    print('Genetic Time:', timeit.default_timer() - start_time2)

if __name__ == "__main__":
    main()

    
