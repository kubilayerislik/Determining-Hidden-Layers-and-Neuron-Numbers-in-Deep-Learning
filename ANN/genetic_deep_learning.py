import logging
import random
from functools import reduce
from operator import add

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)


def get_data(dataset_path):
    batch_size = 64
    input_shape = (910,)

    data = read_excel(dataset_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    train_data, test_data = train_test_split(scaled_data,
                                             train_size=0.630308380,
                                             test_size=0.157577093,
                                             random_state=42)
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0:1]
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0:1]

    return batch_size, input_shape, x_train, x_test, y_train, y_test


def compile_model(network, input_shape):
    """Compile a sequential model."""
    nb_layers = network["nb_layers"]
    nb_neurons = network["nb_neurons"]
    activation = network["activation"]
    optimizer = network["optimizer"]

    model = Sequential()

    for _ in range(nb_layers):
        if not model.layers:
            model.add(Dense(nb_neurons,
                            activation=activation,
                            input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer="normal"))

    model.compile(loss="MSE", optimizer=optimizer, metrics=["MSE"])

    return model


def train_and_score(network, dataset_path):
    """Train the model, return test loss."""
    batch_size, input_shape, x_train, x_test, y_train, y_test = get_data(dataset_path)
    model = compile_model(network, input_shape)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=10000,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper],
    )

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]


class Network:
    """Represent a network and let us operate on it.
    Currently only works for an MLP."""

    def __init__(self, nn_param_choices=None):
        """Initialize our network."""
        self.accuracy = 0.0
        self.nn_param_choices = nn_param_choices
        self.network = {}

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties."""
        self.network = network

    def train(self, dataset_path):
        """Train the network and record the accuracy."""
        if self.accuracy == 0.0:
            self.accuracy = train_and_score(self.network, dataset_path)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network MSE: %.4f" % (self.accuracy))


class Optimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(
        self, nn_param_choices, retain=0.4,
        random_select=0.1, mutate_chance=0.2
    ):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network parameters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of networks
        Returns:
            (float): The average accuracy of the population
        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            (list): Two network objects
        """
        children = []
        for _ in range(2):
            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.
        Args:
            network (dict): The network parameters to mutate
        Returns:
            (Network): A randomly mutated network object
        """
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        network.network[mutation] = random.choice(
            self.nn_param_choices[mutation]
        )

        return network

    def evolve(self, pop):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded,
                                       key=lambda x: x[0],
                                       reverse=False)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents


# Setup logging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
    filename="log.txt",
)


def train_networks(networks, dataset_path):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset_path (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset_path)
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average accuracy of a population of networks.
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(generations, population, nn_param_choices, dataset_path):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset_path (str): Dataset to use for training/evaluating
    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" % (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset_path)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info(
            "Generation average: %.2f%%" % (average_accuracy * 100)
        )

        # Evolve, except on the last iteration.
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

    return networks[:5]


def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-' * 80)
    for network in networks:
        network.print_network()


def main():
    generations = 10  # Number of times to evolve the population.
    population = 20  # Number of networks in each generation.
    dataset_path = "dataset.xlsx"  # Dataset file path.

    nn_param_choices = {
        "nb_neurons": [64, 128, 256],
        "nb_layers": [1, 2, 3, 4],
        "activation": ["relu", "elu"],
        "optimizer": ["rmsprop", "adam", "sgd"],
    }

    logging.info("***Evolving %d generations with population %d***" % (generations, population))

    # Generate the networks.
    networks = generate(generations,
                        population,
                        nn_param_choices,
                        dataset_path)

    # Print the top networks.
    print_networks(networks)


if __name__ == "__main__":
    main()
