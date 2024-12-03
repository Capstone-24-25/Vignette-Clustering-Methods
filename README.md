# Vignette-Group-11
Vignette on comparing three clustering methods (k-nearest-neighbors, spectral, agglomerative); created as a class project for PSTAT197A in Fall 2024.

### Contributors
- Kasturi Sharma
- Nikhil Kuniyil
- Daniel Yan
- Johnson Sy Leung
- Lucas Joseph


### Vignette Abstract

For this project, we are showcasing how clustering methods can be applied to various datasets. For our specific vignette, we chose the user behavior dataset that lists a person's phone model, operating system, app usage, battery life, etc. 
Our objective for this vignette is compare three different clustering methods -- k-nearest neighbors, spectral, and agglomerative -- to group users based on similarities on their usage patterns. We will go through how to implement each of three methods by using this data set. 

### Repository Contents 
-   `data` contains

    -   `user_behavior_dataset.csv` raw data 

-   `scripts` contains

    -   `knn_script.py` script used to generate the processed data

    -   `spectral_script.py` concise version of in-class codes used to replicate published analysis

    -   `agglomerative_script.py` concise version of in-class codes used to replicate published analysis


-   `results` contains a report template `primary-vignette.ipynb` and `primary-vignette.html`

### Reference List
For further information on clustering methods implemented on Python, here are some links that may help!
1. https://scikit-learn.org/1.5/modules/clustering.html
2. https://www.geeksforgeeks.org/ml-spectral-clustering/
3. https://medium.com/@khalidassalafy/agglomerative-hierarchical-clustering-a-study-and-implementation-in-python-fddfdb6a7a64