import torch
import pygad.torchga
import pygad
import numpy
from model import Image_classification
def fitness_func(ga_instanse, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    predictions = pygad.torchga.predict(model=model, 
                                        solution=solution, 
                                        data=data_inputs)

    solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

model=Image_classification()
torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=40)

loss_function = torch.nn.CrossEntropyLoss()

# Data inputs
data_inputs = torch.from_numpy(numpy.load("dataset_inputs.npy")).float()
data_inputs = data_inputs.reshape((data_inputs.shape[0], data_inputs.shape[3], data_inputs.shape[1], data_inputs.shape[2]))

# Data outputs
data_outputs = torch.from_numpy(numpy.load("dataset_outputs.npy")).long()

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 200 # Number of generations.
num_parents_mating = 20 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       #fitness_batch_size=1, #suppose to create the batches and applied the fitness function on them, depend on sol_per_pop
                       fitness_func=fitness_func,
                       #K_tournament=10,
                       #parent_selection_type="tournament",
                      # crossover_type="scattered",
                      # mutation_type="random",
                      #save_solutions=True,
                       #keep_parents=5,
                       on_generation=callback_generation)


# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

predictions = pygad.torchga.predict(model=model, 
                                    solution=solution, 
                                    data=data_inputs)
# print("Predictions : \n", predictions)

# Calculate the crossentropy for the trained model.
print("Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

# Calculate the classification accuracy for the trained model.
accuracy = torch.true_divide(torch.sum(torch.max(predictions, axis=1).indices == data_outputs), len(data_outputs))
print("Accuracy : ", accuracy.detach().numpy())