"""Code to handle a digital twin data structure."""


class SlabDigitalTwin:
    """A class for a digital twin of a slab system."""

    available_statuses = [
        "answer",
        "symbols",
        "bulk",
        "millers",
        "cell_shift",
        "site_placement",
    ]

    def __init__(self, computational_object: dict = {}, info: dict = {}):
        """Initialize self.

        Status indicates the most recent completed stage of creation
        (i.e. bulk, millers, cell_shift,...). The computational object
        is the object underlying self."""

        self.computational_object = computational_object

    @property
    def row(self):
        """Return the database row associated with self."""
        row = {}
        for k in self.available_statuses:
            if k in self.computational_object:
                row[k] = self.computational_object[k]
            else:
                row[k] = None

    @property
    def status(self):
        """Return the curent state of creation."""
        max_idx = -1
        for i, k in enumerate(self.available_statuses):
            idx = self.available_statuses.index(k)
            if idx > max_idx:
                max_idx = idx
        return self.available_statuses[max_idx]

    @property
    def completed(self):
        """Return whether creation is completed."""
        for k in self.available_statuses:
            if (
                k not in self.computational_object
                or self.computational_object[k] in None
            ):
                return False
        # else:
        return True
