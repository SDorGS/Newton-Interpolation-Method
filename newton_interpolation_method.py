import numpy as np

# Given data
x = np.array([1,2,3,4,5,6,7,8])
y = np.array([3010, 3424, 3802, 4105, 4472, 4771, 5051, 5315])

# Calculate the first, second, third, and fourth order differences
def calculate_differences(y):
    diffs = [y]
    while len(diffs[-1]) > 1:
        diffs.append(np.diff(diffs[-1]))
    return diffs

# Check for consistent alternating signs
def has_consistent_alternating_signs(diffs):
    signs = np.sign(diffs)
    return all(signs[i] * signs[i + 1] < 0 for i in range(len(signs) - 1))

# Calculate differences for original data
diffs_original = calculate_differences(y)

# Determine the order at which to stop based on consistent alternating signs
stop_order = None
for i in range(1, len(diffs_original)):
    if has_consistent_alternating_signs(diffs_original[i]):
        stop_order = i
        break

# Find the index of the largest change in the second order differences
index = np.argmax(np.abs(diffs_original[2])) + 1

# Print the original value
print(f"Original value at x={x[index]}: y={y[index]}")

# Create a new array for the corrected values
y_corrected = y.copy()

# Correct the value by averaging its neighbors
y_corrected[index] = (y_corrected[index - 1] + y_corrected[index + 1]) / 2

# Print the corrected value
print(f"Corrected value at x={x[index]}: y={y_corrected[index]}")

# Calculate differences for corrected data
diffs_corrected = calculate_differences(y_corrected)

# Print the differences for corrected data
diffs_corrected_output = {f"Order {i} differences": diff for i, diff in enumerate(diffs_corrected)}



original_table = {
    'x': x,
    'y (Original)': y,
    'y (Corrected)': y_corrected,
    'Δy': diffs_original[1] if stop_order is None or stop_order >= 1 else None,
    'Δ²y': diffs_original[2] if stop_order is None or stop_order >= 2 else None,
    'Δ³y': diffs_original[3] if stop_order is None or stop_order >= 3 else None,
    'Δ⁴y': diffs_original[4] if stop_order is None or stop_order >= 4 else None
}

# Print the corrected table
corrected_table = {
    'x': x,
    'y (Original)': y,
    'y (Corrected)': y_corrected,
    'Δy': diffs_corrected[1] if stop_order is None or stop_order >= 1 else None,
    'Δ²y': diffs_corrected[2] if stop_order is None or stop_order >= 2 else None,
    'Δ³y': diffs_corrected[3] if stop_order is None or stop_order >= 3 else None,
    'Δ⁴y': diffs_corrected[4] if stop_order is None or stop_order >= 4 else None
}

print(original_table)

print(diffs_corrected_output)

print(corrected_table)
print(f"Stop order: {stop_order}")
