import os
import zipfile
import torch
import io

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn
import random
from models.resnet import resnet18

model = torch.load('qresnet18-0-7.pth')

# model.eval()

def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        shuffle= False,
        batch_size=128,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    return dataloader

transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)),
        ]
    )
dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

evens = list(range(0, len(dataset), 10))
trainset_1 = torch.utils.data.Subset(dataset, evens)

data = val_dataloader()


import timeit
correct = 0
total = 0

model.eval()
start_time = timeit.default_timer()
with torch.no_grad():
   for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
      images, labels = images.to("cpu"), labels.to("cpu")
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      acc = 100 * correct / total
print(timeit.default_timer() - start_time)
print('Accuracy of the network on the 10000 test images: %.4f %%' % (
   100 * correct / total))


def testt(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for iteraction, (images, labels) in enumerate(data):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return(100 * correct / total)


model = model.to('cuda')
result = testt(model)
print(result)


layer_names2 = [name for name in model.state_dict().keys() if 'weight._data' in name]
num_weights_per_layer2 = {name: model.state_dict()[name].numel() for name in layer_names2}
n = 0
for _ in layer_names2:
  n = n + num_weights_per_layer2[_]
print(n)


import random
import numpy as np
from deap import base, creator, tools, algorithms
import struct


model = model.to('cuda')
# Define the evaluation function
def evaluate(individual):
    # print("Evaluating individual:", individual)
    model_copy = torch.load('qresnet18-0-7.pth')
    model_copy = model_copy.to('cuda')

###################################################################

    state_dict = model_copy.state_dict()
    for layer_name, weight_idx in individual:
        weight = state_dict[layer_name].view(-1).to('cuda')
        # print(weight)
        # weight[weight_idx] += 0.01  # Perturb the weight slightly

        bit_position = random.randint(0, 2)
        #print(f"bit position:{iibit}")

        #print("before:", weight[weight_idx])
        flipped_value = weight[weight_idx] ^ (1 << bit_position)
        weight[weight_idx] = flipped_value
        #print("after:", weight[weight_idx])


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
print("ind size:", ind_size)

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

if __name__ == "__main__":
    main()

