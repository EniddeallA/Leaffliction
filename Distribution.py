import os
import sys
import matplotlib.pyplot as plt
# Directory structure example:
# Apple
# ├── Apple_Black_rot
# │   ├── Augmented
# │   │   ├── image (1)_Blur.jpg
# │   │   ├── image (1)_Flip.jpg
# │   │   ├── ...
# │   ├── image (1).jpg
# │   ├── image (2).jpg
# │   ├── ...
# ├── Apple_Healthy
# │   ├── Augmented
# │   │   ├── image (1)_Blur.jpg
# │   │   ├── image (1)_Flip.jpg
# │   │   ├── ...
# │   ├── image (1).jpg
# │   ├── image (2).jpg
# │   ├── ...
# ├── ...
# Grape
# ├── ...

# This function will take a path to a directory
# of a folder of different plant diseases
# then plot the distribution of the images in the
# directory in a pie chart and bar chart


def Distribution(path: str):
    # Initialize a counter for the diseases
    disease_counter = {}

    # Traverse the directory
    for root, dirs, files in os.walk(path):
        # Skip the root directory itself
        if root == path:
            continue

        # Check if we are in a disease folder (not in the augmented subfolder)
        if 'Augmented' in root:
            continue

        disease_name = os.path.basename(root)
        if disease_name not in disease_counter:
            disease_counter[disease_name] = 0

        # Count the images in the current directory
        for file in files:
            if file.lower().endswith('.jpg'):
                disease_counter[disease_name] += 1
        # Count the images in the Augmented directory if exists
        augmented_path = os.path.join(root, 'Augmented')
        if os.path.exists(augmented_path):
            for file in os.listdir(augmented_path):
                if file.lower().endswith('.jpg'):
                    disease_counter[disease_name] += 1

    labels = list(disease_counter.keys())
    sizes = list(disease_counter.values())

    # Plot pie chart
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Pie Chart')

    # Plot bar chart
    plt.subplot(1, 2, 2)
    plt.bar(labels, sizes, color='skyblue')
    plt.xlabel('Plant Disease')
    plt.ylabel('Number of Images')
    plt.title('Bar Chart')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    Distribution(sys.argv[1])
