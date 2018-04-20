# Metrics for Comparing Neuronal Tree Structures

--

**Goal**: To vectorize neuron structures based on features and compare them using ideas from topological data analysis.

![alt text][overview]

## Method

#### Step 1: Persistence diagram summary

- a way to characterize and identify meaningful features of a structure
- Relies on *descriptor functions*, functions that encode some sort of information about the structure, e.g. Euclidean distance from a point in the tree
- Heres an example that uses the start and end of a subtree as the descriptor function $f$:

![alt text][persistence-diagram]

- Figure (A) is the tree $T$
- Figure (B) is the persistence diagram $D$ associated with tree $T$, which is just the plot of the descriptor function $f$
- Figure (C) compares two different persistence diagrams: $D$ and $D'$

#### Step 2: Vectorization of persistence diagram summaries

![alt text][vectorization]

- The advantage to the persistence feature vectorization framework is the generality of the descriptor function $f$
	- i.e. we can use many descriptor functions all encoding different "morphometric measurements"
- We simply obtain the persistence diagram $D_i$ from each descriptor function, and convert each $D_i$ into feature vector $v_i$ and concatenate them into a single vector









[overview]: http://journals.plos.org/plosone/article/figure/image?id=10.1371/journal.pone.0182184.g001&size=large

[persistence-diagram]: http://journals.plos.org/plosone/article/figure/image?id=10.1371/journal.pone.0182184.g002&size=large

[vectorization]: http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182184.g004&type=large



