from HyperbolicGraphSimulation import HyperbolicGraphSimulation
import random
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
from python_tsp.exact import solve_tsp_branch_and_bound
from typing import List, Optional, TextIO, Tuple
from python_tsp.utils import (
    setup_initial_solution,
)


def filter_lists(list1, list2, threshold):
    """
    Check if each element in list1 is below the threshold, leaving only the element that satisfies the condition.
    Parameters
    ----------
    list1, list2
        Lists to be classified.
    threshold
        value to classify lists.
    Returns
    ----------
    result_list1, result_list2
        Classified lists.
    """
    filtered_coords = [(x, y) for x, y in zip(list1, list2) if x <= threshold]
    # split the results
    result_list1, result_list2 = zip(*filtered_coords)

    return result_list1, result_list2


def _cycle_to_successors(cycle: List[int]) -> List[int]:
    """
    Convert a cycle representation to successors representation.
    Parameters
    ----------
    cycle
        A list representing a cycle.
    Returns
    -------
    List
        A list representing successors.
    """
    successors = cycle[:]
    n = len(cycle)
    for i, _ in enumerate(cycle):
        successors[cycle[i]] = cycle[(i + 1) % n]
    return successors


def _successors_to_cycle(successors: List[int]) -> List[int]:
    """
    Convert a successors representation to a cycle representation.
    Parameters
    ----------
    successors
        A list representing successors.
    Returns
    -------
    List
        A list representing a cycle.
    """
    cycle = successors[:]
    j = 0
    for i, _ in enumerate(successors):
        cycle[i] = j
        j = successors[j]
    return cycle


def _minimizes_hamiltonian_path_distance(
    tabu: np.ndarray,
    iteration: int,
    successors: List[int],
    ejected_edge: Tuple[int, int],
    distance_matrix: np.ndarray,
    hamiltonian_path_distance: float,
    hamiltonian_cycle_distance: float,
) -> Tuple[int, int, float]:
    """
    Minimize the Hamiltonian path distance after ejecting an edge.
    Parameters
    ----------
    tabu
        A NumPy array for tabu management.
    iteration
        The current iteration.
    successors
        A list representing successors.
    ejected_edge
        The edge that was ejected.
    distance_matrix
        A NumPy array representing the distance matrix.
    hamiltonian_path_distance
        The Hamiltonian path distance.
    hamiltonian_cycle_distance
        The Hamiltonian cycle distance.
    Returns
    -------
    Tuple
        The best c, d, and the new Hamiltonian path distance found.
    """
    a, b = ejected_edge
    best_c = c = last_c = successors[b]
    path_cb_distance = distance_matrix[c, b]
    path_bc_distance = distance_matrix[b, c]
    hamiltonian_path_distance_found = hamiltonian_cycle_distance

    while successors[c] != a:
        d = successors[c]
        path_cb_distance += distance_matrix[c, last_c]
        path_bc_distance += distance_matrix[last_c, c]
        new_hamiltonian_path_distance_found = (
            hamiltonian_path_distance
            + distance_matrix[b, d]
            - distance_matrix[c, d]
            + path_cb_distance
            - path_bc_distance
        )

        if (
            new_hamiltonian_path_distance_found + distance_matrix[a, c]
            < hamiltonian_cycle_distance
        ):
            return c, d, new_hamiltonian_path_distance_found

        if (
            tabu[c, d] != iteration
            and new_hamiltonian_path_distance_found
            < hamiltonian_path_distance_found
        ):
            hamiltonian_path_distance_found = (
                new_hamiltonian_path_distance_found
            )
            best_c = c

        last_c = c
        c = d

    return best_c, successors[best_c], hamiltonian_path_distance_found


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)


def solve_tsp_lin_kernighan(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesperson Problem using the Lin-Kernighan algorithm.
    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j
    x0
        Initial permutation. If not provided, it starts with a random path.
    log_file
        If not `None`, creates a log file with details about the whole
        execution.
    verbose
        If true, prints algorithm status every iteration.
    Returns
    -------
    Tuple
        A tuple containing the Hamiltonian cycle and its distance.
    References
    ----------
    Éric D. Taillard, "Design of Heuristic Algorithms for Hard Optimization,"
    Chapter 5, Section 5.3.2.1: Lin-Kernighan Neighborhood, Springer, 2023.
    """
    hamiltonian_cycle, hamiltonian_cycle_distance = setup_initial_solution(
        distance_matrix=distance_matrix, x0=x0
    )
    num_vertices = distance_matrix.shape[0]
    vertices = list(range(num_vertices))
    iteration = 0
    improvement = True
    tabu = np.zeros(shape=(num_vertices, num_vertices), dtype=int)

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    while improvement:
        iteration += 1
        improvement = False
        successors = _cycle_to_successors(hamiltonian_cycle)

        # Eject edge [a, b] to start the chain and compute the Hamiltonian
        # path distance obtained by ejecting edge [a, b] from the cycle
        # as reference.
        a = int(distance_matrix[vertices, successors].argmax())
        b = successors[a]
        hamiltonian_path_distance = (
            hamiltonian_cycle_distance - distance_matrix[a, b]
        )

        while True:
            ejected_edge = a, b

            # Find the edge [c, d] that minimizes the Hamiltonian path obtained
            # by removing edge [c, d] and adding edge [b, d], with [c, d] not
            # removed in the current ejection chain.
            (
                c,
                d,
                hamiltonian_path_distance_found,
            ) = _minimizes_hamiltonian_path_distance(
                tabu,
                iteration,
                successors,
                ejected_edge,
                distance_matrix,
                hamiltonian_path_distance,
                hamiltonian_cycle_distance,
            )

            # If the Hamiltonian cycle cannot be improved, return
            # to the solution and try another ejection.
            if hamiltonian_path_distance_found >= hamiltonian_cycle_distance:
                break

            # Update Hamiltonian path distance reference
            hamiltonian_path_distance = hamiltonian_path_distance_found

            # Reverse the direction of the path from b to c
            i, si, successors[b] = b, successors[b], d
            while i != c:
                successors[si], i, si = i, si, successors[si]

            # Don't remove again the minimal edge found
            tabu[c, d] = tabu[d, c] = iteration

            # c plays the role of b in the next iteration
            b = c

            msg = (
                f"Current value: {hamiltonian_cycle_distance}; "
                f"Ejection chain: {iteration}"
            )
            _print_message(msg, verbose, log_file_handler)

            # If the Hamiltonian cycle improves, update the solution
            if (
                hamiltonian_path_distance + distance_matrix[a, b]
                < hamiltonian_cycle_distance
            ):
                improvement = True
                successors[a] = b
                hamiltonian_cycle = _successors_to_cycle(successors)
                hamiltonian_cycle_distance = (
                    hamiltonian_path_distance + distance_matrix[a, b]
                )

    if log_file_handler:
        log_file_handler.close()

    return hamiltonian_cycle, hamiltonian_cycle_distance


def graph_to_distance_matrix(G):
    size = len(G)
    dist_matrix = np.zeros((size, size))
    for i, u in enumerate(G.nodes()):
        for j, v in enumerate(G.nodes()):
            if i != j:
                dist_matrix[i][j] = G[u][v]['weight'] if G.has_edge(u, v) else 1500
    return dist_matrix


def hyperbolic_graph_generator(n, k, alpha):
    """
    Generate hyperbolic graph using networkit.
    :param n: The number of nodes of graph
    :param k: Average degree of nodes
    :param alpha: Negative curvature
    :return: Connected hyperbolic graph G
    """
    # set up simulator
    simulator = HyperbolicGraphSimulation()

    # generate graph
    G = simulator.generateGraph(n, k, alpha)

    return G


def connected_generate(n, k, p, graph_type, alpha=3, tries=1000):
    """
    Generate connected graphs
    :param n: The number of nodes of graph
    :param k: Average degree of nodes
    :param p: rewiring probability
    :param graph_type: string, 'nw'(=non watts-strogatz) or 'hyper'(hyperbolic) or 'ws'(watts-strogatz)
    :param tries: Maximum tries to get connected graph
    :param seed: seed
    :return: graph G
    """
    for i in range(tries):
        nodes = list(range(n))
        if graph_type == 'nw':
            G = nx.erdos_renyi_graph(n, k / (n - 1))

            # check connectivity of graph
            while not nx.is_connected(G):
                G = nx.erdos_renyi_graph(n, k / (n - 1))

        elif graph_type == 'hyper':
            G = hyperbolic_graph_generator(n, k, alpha)

            # check connectivity of graph
            while not nx.is_connected(G):
                G = hyperbolic_graph_generator(n, k, alpha)

        else:
            G = nx.connected_watts_strogatz_graph(n, k, p)
            for (u, v, w) in G.edges(data=True):
                w['weight'] = random.randint(1, 10)
            return G

        for j in range(1, k // 2 + 1):  # outer loop is neighbors
            # inner loop in node order
            edges = G.edges()
            for edge in edges:
                u, v = edge
                if random.random() < p:
                    w = random.choice(nodes)
                    # Enforce no self-loops or multiple edges
                    while w == u or G.has_edge(u, w):
                        w = random.choice(nodes)
                        if G.degree(u) >= n - 1:
                            break  # skip this rewiring
                    else:
                        G.remove_edge(u, v)
                        G.add_edge(u, w)
                    if nx.is_connected(G) == False:
                        G.remove_edge(u, w)
                        G.add_edge(u, v)
        if nx.is_connected(G):
            for (u, v, w) in G.edges(data=True):
                w['weight'] = random.randint(1, 10)
            return G
        raise nx.NetworkXError("Maximum number of tries exceeded")

def solve_tsp_python_tsp_sa(dist_matrix):
    permutation, distance = solve_tsp_simulated_annealing(dist_matrix)
    return permutation, distance

def solve_tsp_lin_kernighan_tsp(dist_matrix):
    permutation, distance = solve_tsp_lin_kernighan(dist_matrix)
    return permutation, distance

def solve_tsp_local_search_tsp(dist_matrix):
    permutation, distance = solve_tsp_local_search(dist_matrix)
    return permutation, distance

def solve_tsp_branch_and_bound_tsp(dist_matrix):
    permutation, distance = solve_tsp_branch_and_bound(dist_matrix)
    return permutation, distance

def print_results(name, solution, distance, time_taken):
    print(f"\n{name} - Solution:", solution)
    print(f"{name} - Total Distance:", distance)
    print(f"{name} - Computation Time:", time_taken, "seconds")


def main(cities, nn, graph_type, alpha):
    # Repeats calculation for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)
            # Generate a distance matrix for the graph and save it as an array with indexing
            # mat[rewiring probabilities][trials]
            mat = []
            # Repeats calculation for the rewiring probabilities
            rewiring_probs = [1, 0.1, 0.01, 0.001, 0.0001]
            for prob in rewiring_probs:
                print('rewiring_probs =', prob)
                mat_trial = []
                # Repeats as many times as the set num_trials
                num_trials = 100
                for _ in range(num_trials):
                    print(_)
                    # Generate only connected graphs
                    G = connected_generate(num_cities, nearest_neighbors, prob, graph_type, alpha=alpha)
                    # Convert nx.Graph data to matrix
                    distance_matrix = graph_to_distance_matrix(G)

                    mat_trial.append(distance_matrix.tolist())

                mat.append(mat_trial)

            # Save generated matrix in the dictionary
            result = {}
            result['distance_matrix'] = mat

            # save dictionary
            with open(f'{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'w') as file:
                json.dump(result, file)


# regenerate graphs in which concorde optimization results do not satisfy constraints
def main2(cities, nn, graph_type, alpha):
    rewiring_probs = [1, 0.1, 0.01, 0.001, 0.0001]
    # Repeats calculation for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)
            # read result saved as json file
            with open(f'{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)
            # read concorde optimization result
            with open(f'{graph_type}_{alpha}_concorde opt data_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result2 = json.load(file)
            # Gets stored distance matrix data
            mat = result['distance_matrix']
            # Save only valid graphs and concorde optimization results
            distance_matrix = []
            con_result = []
            # and generate new matrix set to be tested
            mat_new = []
            for p_value in range(1, 5):
                # Leave only if the optimized value is below the threshold
                threshold_value = 1500
                len_result2 = len([x for x in result2[p_value] if x < threshold_value])
                if len_result2 != 0:
                    result_matrix1, result_matrix2 = filter_lists(result2[p_value], mat[p_value], threshold_value)
                else:
                    result_matrix1 = []
                    result_matrix2 = []
                print(len(result_matrix1))
                distance_matrix.append(result_matrix2)
                con_result.append(result_matrix1)
                # generate new matrix set
                mat_trial = []
                num_trials = 3000
                for _ in range(num_trials):
                    print(_)
                    G = connected_generate(num_cities, nearest_neighbors, rewiring_probs[p_value], graph_type, alpha=alpha)
                    aa = graph_to_distance_matrix(G)

                    mat_trial.append(aa.tolist())

                mat_new.append(mat_trial)

            # Save modified data and new matrix set in the dictionary
            result1 = {}
            result1['con_result'] = con_result
            result1['distance_matrix'] = distance_matrix
            result1['new_mat_set'] = mat_new

            # save dictionary to a new json file
            with open(f'{graph_type}2_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'w') as file:
                json.dump(result1, file)


# regenerate graphs in which concorde optimization results do not satisfy constraints again
def main3(cities, nn, graph_type, alpha):
    rewiring_probs = [1, 0.1, 0.01, 0.001, 0.0001]
    # Repeats calculation for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)
            # read result saved as json file
            with open(f'{graph_type}3_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)
            # read concorde optimization result
            with open(f'{graph_type}_{alpha}_concorde opt data_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result2 = json.load(file)
            # Gets stored data
            distance_matrix = result['distance_matrix']
            con_result = result['con_result']
            mat_new = result['new_mat_set']
            # Save only valid graphs and concorde optimization results from the new matrix set
            new_mat = []
            for p_value in range(5):
                # Leave only if the optimized value is below the threshold
                threshold_value = 1500
                if len(result2[p_value]) != 0:
                    if len([element for element in result2[p_value] if element < threshold_value]) != 0:
                        result_matrix1, result_matrix2 = filter_lists(result2[p_value], mat_new[p_value],
                                                                      threshold_value)
                        distance_matrix[p_value].extend(result_matrix2)
                        con_result[p_value].extend(result_matrix1)
                print(len(con_result[p_value]))
                # if the number of valid graphs exceeds 100, leave only 100 matrices
                # and don't generate new matrix
                if len(con_result[p_value]) >= 100:
                    distance_matrix[p_value] = distance_matrix[p_value][:100]
                    con_result[p_value] = con_result[p_value][:100]

                    new_mat.append([])
                # if the number of valid graphs is less than 100, generate new matrix set
                elif len(con_result[p_value]) < 100:
                    mat_trial = []
                    num_trials = 1000
                    for _ in range(num_trials):
                        G = connected_generate(num_cities, nearest_neighbors, rewiring_probs[p_value], graph_type, alpha=alpha)
                        aa = graph_to_distance_matrix(G)

                        mat_trial.append(aa.tolist())

                    new_mat.append(mat_trial)

            # Save modified data and new matrix set in the dictionary
            result1 = {}
            result1['con_result'] = con_result
            result1['distance_matrix'] = distance_matrix
            result1['new_mat_set'] = new_mat

            # save dictionary
            with open(f'{graph_type}3_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'w') as file:
                json.dump(result1, file)


# add optimized result of SA, lk
def main4(cities, nn, graph_type, alpha):
    # Repeats calculation for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)

            # read result saved as json file
            with open(f'{graph_type}3_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)

            distance_matrix = result['distance_matrix']
            con_result = result['con_result']

            num_trials = 100

            # Calculate optimization results of Simulated Annealer(SA) and lin-kernighan algorithm
            sa_result = []      # optimized energy
            lk_result = []

            sa_list = []        # optimized vector
            lk_list = []

            for p_value in range(5):
                sa_result_trial = []
                lk_result_trial = []

                sa_list_trial = []
                lk_list_trial = []

                for trial in range(num_trials):
                    print('nn =', nearest_neighbors, ', num_cities =', num_cities, ', p =', p_value, ', trial =', trial)

                    sa_repeat = []
                    lk_repeat = []
                    sa_re = []
                    lk_re = []
                    for repeat in range(5):
                        # Simulated Annealing
                        sa_opt, sa_distance = solve_tsp_python_tsp_sa(np.array(distance_matrix[p_value][trial]))
                        sa_repeat.append(sa_opt)
                        sa_re.append(sa_distance)

                        # Lin-Kernighan
                        lk_opt, lk_distance = solve_tsp_lin_kernighan_tsp(np.array(distance_matrix[p_value][trial]))
                        lk_repeat.append(lk_opt)
                        lk_re.append(lk_distance)

                    min_index1 = sa_re.index(min(sa_re))
                    sa_result_trial.append(sa_re[min_index1])
                    sa_list_trial.append(sa_repeat[min_index1])

                    min_index2 = lk_re.index(min(lk_re))
                    lk_result_trial.append(lk_re[min_index2])
                    lk_list_trial.append(lk_repeat[min_index2])

                sa_result.append(sa_result_trial)
                lk_result.append(lk_result_trial)

                sa_list.append(sa_list_trial)
                lk_list.append(lk_list_trial)

            # Save datas in the dictionary
            result1 = {}
            result1['con_result'] = con_result
            result1['sa_result'] = sa_result
            result1['lk_result'] = lk_result
            result1['distance_matrix'] = distance_matrix
            result1['sa_opt'] = sa_list
            result1['lk_opt'] = lk_list

            # save dictionary
            with open(f'dataset_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'w') as file:
                json.dump(result1, file)


# plot ratio satisfying constraints and error compared to concorde results
def main5(cities, nn, graph_type, alpha):
    rewiring_probs = [1, 0.1, 0.01, 0.001, 0.0001]
    # Repeats calculation for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)

            # read result saved as json file
            with open(f'dataset_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)

            sa_result = result['sa_result']
            lk_result = result['lk_result']
            con_result = result['con_result']

            sa_success_counts = []      # error compared to concorde
            lk_success_counts = []

            sa_inter_counts = []        # ratio satisfying constraints
            lk_inter_counts = []

            # Calculate approximation ratio of SA, lk compared to concorde
            for prob in range(5):
                errors = np.array(sa_result[prob]) - np.array(con_result[prob])     # error = (SA result) - (concorde result)
                errors = errors[np.array(sa_result[prob]) <= 1500]      # Only when the constraints are satisfied are considered
                con_a = np.array(con_result[prob])[np.array(sa_result[prob]) <= 1500]
                if len(con_a) == 0:
                    sa_success = np.array([])
                else:
                    sa_success = (errors / con_a) + 1       # divide the error by the result of concorde
                sa_success_counts.append(sa_success.tolist())

                errors = np.array(lk_result[prob]) - np.array(con_result[prob])
                errors = errors[np.array(lk_result[prob]) <= 1500]
                con_aa = np.array(con_result[prob])[np.array(lk_result[prob]) <= 1500]
                if len(con_aa) == 0:
                    lk_success = np.array([])
                else:
                    lk_success = (errors / con_aa) + 1
                lk_success_counts.append(lk_success.tolist())

                # Calculate ratio of results satisfying constraints
                sa_inter = sum(x <= 1500 for x in sa_result[prob]) / 100
                lk_inter = sum(x <= 1500 for x in lk_result[prob]) / 100

                sa_inter_counts.append(sa_inter)
                lk_inter_counts.append(lk_inter)

            # Plotting results
            fig, ax1 = plt.subplots(figsize=(7, 5))
            plt.title(f'k{nearest_neighbors} n{num_cities}', fontsize=14)

            # Set x-axis
            plt.xlabel('Rewiring Probability (Log Scale)', fontsize=14, labelpad=12, fontweight='bold')
            plt.xscale('log')  # Setting x-axis to log scale

            # Set y-axis1 data
            ax1.set_ylabel('ratio satisfying constraints', fontsize=14, labelpad=12, fontweight='bold')
            ax1.plot(rewiring_probs, sa_inter_counts, label='ratio of SA satisfying constraint', linestyle='--', color='darkorange')
            ax1.plot(rewiring_probs, lk_inter_counts, label='ratio of lk satisfying constraint', linestyle='--', color='cornflowerblue')
            ax1.set_ylim(0, 1)
            ax1.tick_params(labelsize=16)

            # Set y-axis2 data
            ax2 = ax1.twinx()
            ax2.set_ylabel('approximation ratio', fontsize=14, labelpad=12, fontweight='bold')
            widths = [0.7, 0.07, 0.007, 0.0007, 0.00007]        # Adjust the width of the boxplot to fit the log scale
            bplot1 = ax2.boxplot(sa_success_counts, meanline='True', vert=True, patch_artist=True, widths=widths, positions=rewiring_probs, medianprops={'color': 'Red'})
            for patch in bplot1['boxes']:
                patch.set_facecolor('darkorange')
            bplot2 = ax2.boxplot(lk_success_counts, meanline='True', vert=True, patch_artist=True, widths=widths, positions=rewiring_probs, medianprops={'color': 'Blue'})
            for patch in bplot2['boxes']:
                patch.set_facecolor('cornflowerblue')
            ax2.set_ylim(0.5, 1.8)
            ax2.tick_params(labelsize=16)

            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9, wspace=0.3)

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            # Save the plotted image
            plt.savefig(f'fig_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}')


def swi(G, niter=20, nrand=10, seed=None):
    """Returns the small-world index(SWI) of a graph

    The small-world coefficient of a graph G is:

    swi = (L-Ll)(C-Cr)/(Lr-Ll)(Cl-Cr)

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    niter: integer (optional, default=5)
        Approximate number of rewiring per edge to compute the equivalent
        random graph.

    nrand: integer (optional, default=10)
        Number of random graphs generated to compute the maximal clustering
        coefficient (Cr) and average shortest path length (Lr).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.


    Returns
    -------
    swi : float
        The small-world index(SWI)
    """
    import numpy as np
    # convert weighted graph G to non-weighted graph
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}

    # Calculate initial average clustering coefficient which potentially will
    # get replaced by higher clustering coefficients from generated lattice
    # reference graphs
    Cl = nx.average_clustering(G)
    Ll = nx.average_shortest_path_length(G)

    niter_lattice_reference = niter
    niter_random_reference = niter * 2

    for _ in range(nrand):
        # Generate random graph
        Gr = nx.random_reference(G, niter=niter_random_reference, seed=seed)
        randMetrics["L"].append(nx.average_shortest_path_length(Gr))
        randMetrics["C"].append(nx.average_clustering(Gr))

        # Generate lattice graph
        Gl = nx.lattice_reference(G, niter=niter_lattice_reference, seed=seed)

        # Replace old clustering coefficient, if clustering is higher in
        # generated lattice reference
        Cl_temp = nx.average_clustering(Gl)
        if Cl_temp > Cl:
            Cl = Cl_temp
            Ll = nx.average_shortest_path_length(Gl)

    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    Lr = np.mean(randMetrics["L"])
    Cr = np.mean(randMetrics["C"])

    swi = (L - Ll) * (C - Cr) / ((Lr - Ll) * (Cl - Cr))

    return swi


# calculate small-worldness(SWI, omega)
def main6(cities, nn, graph_type, alpha):
    # Repeats calculation for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)

            # read result saved as json file
            with open(f'dataset_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)

            distance_matrix = np.array(result['distance_matrix'])
            distance_matrix[distance_matrix == 1500] = 0        # to convert the matrix into a graph, adjust the value where edge does not exist to 0

            num_trials = 20

            # Calculate SWI and omega
            swi_p = []
            omega_p = []

            for p_value in range(5):
                swi_trial = []
                omega_trial = []

                for trial in range(num_trials):
                    print('nn =', nearest_neighbors, ', num_cities =', num_cities, ', p =', p_value, ', trial =', trial)

                    #swi_trial.append(swi(nx.Graph(distance_matrix[p_value][trial])))
                    G = nx.Graph(distance_matrix[p_value][trial])
                    for edge in G.edges():
                        G[edge[0]][edge[1]]['weight'] = 1
                    omega_trial.append(nx.omega(G, niter=20))

                swi_p.append(swi_trial)
                omega_p.append(omega_trial)

            # add small-worldness data into the dictionary
            #result['swi'] = swi_p
            result['omega'] = omega_p

            # save dictionary
            with open(f'dataset_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'w') as file:
                json.dump(result, file)


# plot small-worldness of graph
def main7(cities, nn, graph_type, alpha):
    rewiring_probs = [1, 0.1, 0.01, 0.001, 0.0001]
    # Repeats plotting for the number of cities(vertices) n and the number of nearest neighbors k
    for num_cities in cities:
        print('num_cities =', num_cities)
        for nearest_neighbors in nn:
            print('nearest_neighbors =', nearest_neighbors)

            # read result saved as json file
            with open(f'dataset_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)

            omega = result['omega']

            # Plotting results
            fig, ax1 = plt.subplots(figsize=(7, 5))
            plt.title(f'k{nearest_neighbors} n{num_cities}', fontsize=14)

            # Set x-axis
            plt.xlabel('Rewiring Probability (Log Scale)', fontsize=14, labelpad=12, fontweight='bold')
            plt.xscale('log')  # Setting x-axis to log scale

            # Set y-axis
            ax1.set_ylabel('ratio satisfying constraints', fontsize=14, labelpad=12, fontweight='bold')
            mean_omega = np.mean(omega, axis=1)
            std_omega = np.std(omega, axis=1)
            ax1.errorbar(rewiring_probs, mean_omega, std_omega, label='omega', linestyle='--', color='darkorange')
            ax1.set_ylim(-1, 1)
            ax1.tick_params(labelsize=16)

            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9, wspace=0.3)

            ax1.legend(loc='upper left')

            # Save the plotted image
            plt.savefig(f'fig_{graph_type}_small worldness_n{num_cities}_k{nearest_neighbors}')


def calculate_mean_and_std(lists):
    # Find the Maximum Length
    max_len = max(len(lst) for lst in lists)

    # Add each list to the maximum length
    padded_lists = [lst + [0] * (max_len - len(lst)) for lst in lists]

    # Calculate the mean and standard deviation for each location of the lists
    means = np.mean(padded_lists, axis=0)
    stds = np.std(padded_lists, axis=0)

    return means, stds


# plot degree distribution of graph
def main8(cities, nn, graph_type, alpha):
    rewiring_probs = [1, 0.1, 0.01, 0.001, 0.0001]
    for nearest_neighbors in nn:
        print('nn =', nearest_neighbors)
        for num_cities in cities:
            print('num_cities =', num_cities)

            # read result saved as json file
            with open(f'dataset_{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json', 'r') as file:
                result = json.load(file)

            mat = np.array(result['distance_matrix'])
            mat[mat == 1500] = 0
            for rew in range(5):
                degrees = []
                for i in range(100):
                    G = nx.Graph(mat[rew][i])
                    degree_count = nx.degree_histogram(G)
                    degree_count = [100 * x / num_cities for x in degree_count]
                    degrees.append(degree_count)

                mean_degree, std_degree = calculate_mean_and_std(degrees)

                # plotting
                plt.figure(figsize=(7, 5))

                x_positions = np.arange(len(mean_degree))
                plt.bar(x_positions, mean_degree, yerr=std_degree, capsize=5, color='blue', alpha=0.7, align='center')

                labels = x_positions
                plt.xticks(x_positions, labels)

                plt.title(f'degree distribution of k{nearest_neighbors}, n{num_cities}, p{rewiring_probs[rew]} graph',
                          fontsize=16, fontweight='bold')
                plt.xlabel('degrees', fontsize=14, labelpad=12, fontweight='bold')
                plt.ylabel('ratio of edges', fontsize=14, labelpad=12, fontweight='bold')
                plt.ylim(0, 40)

                plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)

                plt.savefig(f'TSP {graph_type} degree distribution_n{num_cities}_k{nearest_neighbors}_p10^-{rew}')


if __name__ == "__main__":
    number_of_cities = [20]
    average_degree = [4, 5, 6, 7]
    graph_type = 'hyper'        # 'hyper' or 'nw'(non watts strogatz) or 'ws'(watts  strogatz)
    alpha = 1
    # generate graphs
    main(number_of_cities, average_degree, graph_type, alpha)

    '''Solve the TSP problem saved in file '{graph_type}_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json'
        as 'distance matrix' using the concorde solver and save the solution
        to file '{graph_type}_{alpha}_concorde opt data_n{num_cities}_k{nearest_neighbors}.json'.'''

    # regenerate graphs in which concorde optimization results do not satisfy constraints
    # main2(number_of_cities, average_degree, graph_type, alpha)

    '''Solve the TSP problem saved in file '{graph_type}2_{alpha}_TSP_n{num_cities}_k{nearest_neighbors}.json'
        as 'distance matrix' using the concorde solver and save the solution
        to file '{graph_type}_{alpha}_concorde opt data_n{num_cities}_k{nearest_neighbors}.json'.'''

    # regenerate graphs in which concorde optimization results do not satisfy constraints again
    # main3(number_of_cities, average_degree, graph_type, alpha)

    # add optimized results of SA, lk
    # main4(number_of_cities, average_degree, graph_type, alpha)

    # plot ratio satisfying constraints and error with concorde results
    # main5(number_of_cities, average_degree, graph_type, alpha)

    # calculate small-worldness(SWI, omega)
    # main6(number_of_cities, average_degree, graph_type, alpha)

    # plot small-worldness of graph
    # main7(number_of_cities, average_degree, graph_type, alpha)

    # plot degree distribution of graph
    # main8(number_of_cities, average_degree, graph_type, alpha)
