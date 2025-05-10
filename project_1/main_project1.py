

def main():
    """
    Main entry point for the project.
    This function will be called when the script is run directly.
    """
    print("Executing main function...")

    from datasets import load_dataset

    # # Another dataset availabe: sms_spam
    # sms_spam_dataset = load_dataset("sms_spam", split=["train"])[0]
    # print(sms_spam_dataset)
    # # Split the dataset into train and test sets (assuming `dataset` is a Hugging Face Dataset object)
    # sms_spam_dataset_train_test_split = sms_spam_dataset.train_test_split(test_size=0.2, seed=42)
    # test_dataset = sms_spam_dataset_train_test_split['test']
    # print(sms_spam_dataset[:5])
    # """
    # Convert the dataset to a pandas DataFrame
    # """
    # import pandas as pd

    # # Convert the entire dataset to a pandas DataFrame
    # df = pd.DataFrame(sms_spam_dataset)
    # # Display the first 10 rows
    # print(df.head(10))

    # # Or for more detailed information
    # display(df.head(10))  # Works in Jupyter notebooks
    # display(df.describe())  # Works in Jupyter notebooks

    # for entry in dataset.select(range(3)):
    #     sms = entry["sms"]
    #     label = entry["label"]
    #     print(f"label={label}, sms={sms}")

    def load_and_analyze_dataset(dataset_name="imdb", splits=["train", "test"], sample_size=600, seed=42):
        """
        Load, sample, and analyze a dataset for classification tasks.

        Args:
            dataset_name (str): Name of the dataset to load from Hugging Face
            splits (list): List of splits to load (e.g., ["train", "test"])
            sample_size (int): Number of samples to select from each split (None for full dataset)
            seed (int): Random seed for shuffling

        Returns:
            dict: Dictionary containing the dataset splits
        """
        # Load the specified dataset splits
        dataset = {split: ds for split, ds in zip(splits, load_dataset(dataset_name, split=splits))}

        # Sample the dataset if sample_size is provided
        if sample_size is not None:
            for split in splits:
                dataset[split] = dataset[split].shuffle(seed=seed).select(range(sample_size))

        print(f"Loaded {dataset_name} dataset:")
        print(dataset)

        # Count the number of samples per class in each split
        analyze_class_distribution(dataset, splits=splits)

        return dataset

    def analyze_class_distribution(dataset, splits=None):
        """
        Analyze and print the class distribution for each split in a dataset.

        Args:
            dataset (dict): Dictionary mapping split names to dataset objects
            splits (list): List of split names to analyze (if None, analyze all splits in dataset)
        """
        # If no specific splits are provided, analyze all splits in the dataset
        if splits is None:
            splits = dataset.keys()

        for split in splits:
            # Get all unique labels
            all_labels = set([item['label'] for item in dataset[split]])
            total = len(dataset[split])

            print(f"\n{split} split:")
            for label in sorted(all_labels):
                count = sum(1 for item in dataset[split] if item['label'] == label)
                print(f"  Label {label}: {count} ({count / total:.2%})")

        return dataset  # Return dataset to allow for function chaining

    """
    Some conclusions:
    with a 100 dataset also behave very well. A big improvement is seen for accuracy and recall.
    """

    # Example usage:
    splits = ["train", "test"]
    dataset_size = 850
    # https://huggingface.co/datasets/stanfordnlp/imdb
    # dataset_used_to_fine_tuned = load_and_analyze_dataset("imdb", splits, sample_size= dataset_size)

    # To load a different dataset, e.g., sms_spam:
    # TODO: need to fix this one
    # dataset_used_to_fine_tuned = load_and_analyze_dataset("sms_spam", splits=["train"], sample_size= dataset_size)

    # https://huggingface.co/datasets/fancyzhx/ag_news this has 4 classes does not work
    # dataset_used_to_fine_tuned = load_and_analyze_dataset("fancyzhx/ag_news",  sample_size= dataset_size)

    # https://huggingface.co/datasets/stanfordnlp/sst2
    dataset_used_to_fine_tuned = load_and_analyze_dataset("stanfordnlp/sst2", splits=["train", "validation"],
                                                          sample_size=dataset_size)

    dataset_used_to_fine_tuned['test'] = dataset_used_to_fine_tuned['validation']
    validation_data = dataset_used_to_fine_tuned.pop("validation", None)
    analyze_class_distribution(dataset_used_to_fine_tuned, splits=splits)
    # Add your code here
    # For example:
    # 1. Load and preprocess data
    # 2. Create and train a model
    # 3. Evaluate results
    # 4. Save the model

if __name__ == "__main__":
    # This block only executes when the script is run directly
    # (not when imported as a module)
    main()