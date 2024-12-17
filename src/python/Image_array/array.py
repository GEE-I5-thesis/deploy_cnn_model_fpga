# from PIL import Image
# import numpy as np

# # Load the image
# image_path = "232.jpg"  # Replace with your image file path
# image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode

# # Resize the image to 224x224
# image_resized = image.resize((10, 10))

# # Convert the resized image to a NumPy array
# image_array = np.array(image_resized)

# # Normalize pixel values to range [0, 1]
# normalized_array = image_array / 255.0

# # Flatten the array for saving in the desired format
# # Since this is an RGB image, it will have three channels.
# flattened_array = normalized_array.flatten()

# # Format the array into the desired string structure
# array_as_string = ",\n\t\t".join(
#     [", ".join(map(str, flattened_array[i:i + 10])) for i in range(0, len(flattened_array), 10)]
# )
# formatted_string = f"{{\n\t\t{array_as_string}\n}};"

# # Save the formatted string to a text file
# output_file = "output_array.txt"
# with open(output_file, "w") as f:
#     f.write(formatted_string)

# print(f"Array saved to {output_file}")

from PIL import Image
import numpy as np

# Load the image
image_path = "cat1.jpg"  # Replace with your image file path
img = Image.open(image_path)

# Customize size: Specify the target size (width, height)
target_size = (224, 224)  # Replace with the desired size, e.g., (200, 200)

# Resize the image
img = img.resize(target_size)

# Convert the image to RGB (if it's not already in RGB format)
img = img.convert("RGB")

# Convert the image to a NumPy array
img_array = np.array(img)

# Flatten the array to a 1D array (each pixel is now a 3-element RGB array)
flattened_array = img_array.flatten()

# Format the array into the desired string structure with 9 decimal places
array_as_string = ",\n\t\t".join(
    [", ".join([f"{value / 255.0:.9f}" for value in flattened_array[i:i + 10]]) for i in range(0, len(flattened_array), 10)]
)
formatted_string = f"{{\n\t\t{array_as_string}\n}};"

# Save the formatted string to a text file
output_file = "output_array.txt"
with open(output_file, "w") as f:
    f.write(formatted_string)

print(f"Array saved to {output_file}")
