import numpy as np

# Generate a random 224x224x3 array to represent normalized image data
cat_image = np.random.rand(224, 224, 3).astype(np.float32)

# Flatten the array and format as a C++ array
flattened_array = cat_image.flatten()
formatted_array = ",\n    ".join(", ".join(f"{value:.3f}" for value in flattened_array[i:i + 10]) 
                                 for i in range(0, len(flattened_array), 10))

# Create the final string for a C++ array
cpp_array = f"float cat_image[224 * 224 * 3] = {{\n    {formatted_array}\n}};"

# Save to a text file
output_path = "dog.jpg"
with open(output_path, "w") as file:
    file.write(cpp_array)

output_path
