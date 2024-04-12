import os

import pandas as pd
from matplotlib import pyplot as plt


def plot_sdg_distribution(data):
    # Implement a function to plot the distribution of the SDG labels in the dataset
    # The input data is a pandas dataframe with the following columns:
    # - doi (the doi of the sample)
    # - text (the text of the sample)
    # - sdg (the corresponding SDG label)
    # - label (the agreement value that we will use as the target variable)

    # Set the SDG column to int
    data['sdg'] = data['sdg'].astype(int)
    sdg_counts = data['sdg'].value_counts()
    # Exclude SDG 16 from the plot but record it to include it later 
    sdg_16_count = sdg_counts[16]
    sdg_counts = sdg_counts.drop(16)
    sdg_counts.plot(kind='bar')
    plt.title('SDG Distribution')
    # Add SGD 16 to the x-axis labels
    plt.xlabel('SDG Label')
    plt.ylabel('Count')
    # Draw a horizontal one at the lowest value
    plt.axhline(y=sdg_counts.min(), color='r', linestyle='--')
    # Add the meaning of the horizontal line to the legend
    plt.legend(['Minimum Count', 'SDG Count'])

    # Add the count of SDG 16 to the title
    plt.title(f'SDG Distribution (excluding SDG 16, count: {sdg_16_count})')
    
    # Save the plot as a png file
    plt.savefig(os.path.join('data', 'OSDG', 'citing_works_sdg_distribution.png'))


def main():
    OSDG_folder_path = os.path.join('data', 'OSDG')
    citing_works_dataset_path = os.path.join(OSDG_folder_path, 'citing_works_OSDG.csv')
    citing_works_data = pd.read_csv(citing_works_dataset_path, header=0, delimiter = ',', encoding='utf-8')
    plot_sdg_distribution(citing_works_data)


if __name__ == "__main__":
    main()