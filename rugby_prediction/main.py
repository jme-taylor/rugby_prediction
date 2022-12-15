from rugby_prediction.data import load_raw_data


def main():
    raw_match_data = load_raw_data()
    print(len(raw_match_data))
    return None


if __name__ == "__main__":
    main()
