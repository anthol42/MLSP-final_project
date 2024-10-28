from utils import TableBuilder
from pathlib import PurePath

if __name__ == "__main__":
    TableBuilder(["accuracy", "precision", "crossEntropy"], PurePath("./results/resultTable.json")).build()
