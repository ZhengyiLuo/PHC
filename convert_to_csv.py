import numpy as np
import pandas as pd

# Right hand DOF names we're looking for
right_hand_dof_names = [
    'right_wrist_roll_link',
    'right_wrist_pitch_link', 
    'right_wrist_yaw_link',
    'right_thumb_yaw_link',
    'right_thumb_f1_link',
    'right_thumb_f2_link',
    'right_index_finger_f1_link',
    'right_middle_finger_f1_link'
]

# Load the dof data from the text file
with open('113_27.txt', 'r') as f:
    data_str = f.read()
    
    # First, let's try to find the bfs_dof_names to get the mapping
    bfs_start = data_str.find("'bfs_dof_names': [") + len("'bfs_dof_names': [")
    bfs_end = data_str.find("]", bfs_start)
    bfs_str = data_str[bfs_start:bfs_end]
    dof_names = [name.strip().strip("'") for name in bfs_str.split(',')]
    dof_names = [name for name in dof_names if name]  # Remove empty strings
    
    # Extract the dof array
    dof_start = data_str.find("'dof': array([") + len("'dof': array([")
    dof_end = data_str.find("], dtype=float32)", dof_start)
    dof_str = data_str[dof_start:dof_end]
    
    # Convert string representation to numpy array
    dof_rows = [row.strip() for row in dof_str.split('\n')]
    dof_data = []
    for row in dof_rows:
        if row:  # Skip empty lines
            values = [float(x) for x in row.strip('[] ,').split(',') if x.strip()]
            if values:  # Only add non-empty rows
                dof_data.append(values)
    
    dof_array = np.array(dof_data)

# Calculate movement for each DOF
movement = np.ptp(dof_array, axis=0)  # Peak-to-peak range of motion
std_dev = np.std(dof_array, axis=0)   # Standard deviation of motion

# Print analysis of movement
print("\nDOF Movement Analysis:")
print("Index | DOF Name | Range of Motion | Std Dev")
print("-" * 50)
for i, (mov, std) in enumerate(zip(movement, std_dev)):
    dof_name = dof_names[i] if i < len(dof_names) else f"DOF_{i}"
    print(f"{i:3d} | {dof_name:30s} | {mov:10.6f} | {std:10.6f}")

# Find indices of DOFs with significant movement
# Let's consider "significant" as having movement > 1% of the maximum movement
threshold = np.max(movement) * 0.01
moving_dofs = np.where(movement > threshold)[0]

print("\nDOFs with significant movement:")
for idx in moving_dofs:
    dof_name = dof_names[idx] if idx < len(dof_names) else f"DOF_{idx}"
    print(f"Index {idx}: {dof_name} (movement: {movement[idx]:.6f})")

# Find the right hand DOF indices by matching names
right_hand_indices = []
for target_name in right_hand_dof_names:
    matches = [i for i, name in enumerate(dof_names) if target_name in name]
    if matches:
        right_hand_indices.append(matches[0])
    else:
        print(f"Warning: Could not find index for {target_name}")

print("\nIdentified right hand DOF indices:", right_hand_indices)

# Extract the right hand DOFs using the found indices
right_hand_dofs = dof_array[:, right_hand_indices]

# Create a DataFrame
df = pd.DataFrame(right_hand_dofs, columns=right_hand_dof_names)

# Save to CSV
df.to_csv('right_hand_angles.csv', index=False)
print("\nSaved right hand angles to right_hand_angles.csv")
print(f"Shape of data: {df.shape} (frames × DOFs)")
print("\nFirst few rows of the data:")
print(df.head()) 
