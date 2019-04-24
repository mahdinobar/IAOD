from sklearn.cluster import Birch
X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,
            compute_labels=True)
brc.fit(X)

labelsBCHH=brc.predict(X)
