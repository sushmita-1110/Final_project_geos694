# Reflection

## Selected Option
I chose **Option 1: Modify Existing Code**.

I started with an existing Python script that plots Alaska seismic stations using **PyGMT** and **pandas**. Instead of rewriting it, I improved the code so it is cleaner, easier to read, and easier for someone else to run.

## Task 1
For Task 1, I improved formatting, used clearer variable names, and organized the code into smaller parts. I added functions and docstrings so each step of the workflow is easier to understand. I also made regular Git commits instead of one large commit. I kept some PyGMT settings in their original string form because that is the normal PyGMT/GMT syntax. I also added a `README.md` so others can run, and understand the code more easily.

## Task 2
For Task 2, I implemented a class called `AlaskaStationMap`. This made sense because the script keeps using the same settings, like the input file, output file, map region, projection, and title. Putting these into a class made the code more organized.

I did not implement parallel processing because this script only reads one file and makes one map, so it would not really help here. If I had many files to process, I could run one map-making job per file in parallel.

## Task 3
For Task 3, I added a parameter input system using `input()`. This lets the user enter the input and output filenames when the script runs, or just press Enter to use the default names.

I did not implement state saving because this script is simple and fast, so there is not much benefit in saving intermediate results. I also did not build a full testing system, although I did include basic checks like making sure the input file exists and the required columns are present. I did not use branching development for this small script, but I would use branches in a larger project to safely develop new features.
