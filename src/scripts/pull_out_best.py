"""Pull out the best structures."""
import pathlib as path

catalysts = ["CuZnAl", "CuNiZn", "CuFeZn", "NiZnAl", "FeCoZn"]
adsorbates = ["CO2", "*OCHO", "CHOH", "*OHCH3"]

for cat in catalysts:
    for ads in adsorbates:
        adslab_string = f"{cat}_{ads}"
        print(adslab_string)
