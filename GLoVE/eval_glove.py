from explorer_glove import GloVeExplorer

if __name__ == "__main__":
    exp = GloVeExplorer("Bhili")
    exp.nearest_words("पूर्ति", 10)
