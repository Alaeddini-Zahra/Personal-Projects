import itertools
import math

def calculate_distance(fragment1, fragment2):
    # Calculate the distance between two fragments using hamming distance metric
    distance = sum(a != b for a, b in zip(fragment1, fragment2))
    return distance

def calculate_total_distance(fragments, path):
    # Calculate the total distance of a path
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += calculate_distance(fragments[path[i]], fragments[path[i+1]])
    return total_distance

def calculate_identity_percentage(original_sequence, reconstructed_sequence):
    # Calculate the identity percentage between two sequences
    length = len(original_sequence)
    differences = calculate_distance(original_sequence, reconstructed_sequence)
    identity_percentage = ((length - differences) / length) * 100
    return identity_percentage

def tsp(fragments):
    # Generate all possible permutations of the fragments
    all_paths = list(itertools.permutations(range(len(fragments))))

    # Initialize variables to store the best path and its distance
    best_path = None
    best_distance = math.inf

    # Iterate through all paths and calculate their distances
    for path in all_paths:
        distance = calculate_total_distance(fragments, path)
        if distance < best_distance:
            best_distance = distance
            best_path = path

    return best_path, best_distance

# usage
sequence = 'ACGTATGCAGCTACCT'
fragments = [sequence[i:i+4] for i in range(0, len(sequence), 4)]

best_path, best_distance = tsp(fragments)
print("Best Path:", best_path)
print("Best Distance:", best_distance)

# Reconstruct the sequence based on the best path
reconstructed_sequence = ''.join([fragments[i] for i in best_path])
print("Reconstructed Sequence:", reconstructed_sequence)

# Calculate and print the identity percentage
identity_percentage = calculate_identity_percentage(sequence, reconstructed_sequence)
print("Identity Percentage:", identity_percentage)
