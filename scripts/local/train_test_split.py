from src.preprocessing import DatasetProcessor


DATA_PATH = "/home/student/Repositories/lung-cancer-detection/LIDC-IDRI/CT/processed"


def run():
    dp = DatasetProcessor(DATA_PATH)
    # dp.train_test_split()
    # dp.remove_processed_data()
    dp.remove_train_test_data()


if __name__ == "__main__":  
    run()