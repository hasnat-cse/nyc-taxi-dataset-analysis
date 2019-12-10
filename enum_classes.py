import enum

class LocationType(enum.Enum):
    Pickup = 1
    Dropoff = 2


class ClusteringMethod(enum.Enum):
    dbscan = 1
    hdbscan = 2
    optics = 3
    dbscan_hdbscan = 4
    optics_hdbscan = 5
    all = 6
