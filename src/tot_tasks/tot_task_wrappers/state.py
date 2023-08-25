"""A state wrapper for tot tasks."""


class TaskState:
    """State wrapper for tot tasks."""

    def __init__(self, x, task_idx, y="", value=None, depth=0):
        """Store x and ys."""
        self.x = x
        self.y = y
        self.value = value
        self.depth = depth
        self.task_idx = task_idx

    def copy(self):
        """Return a copy of this TaskState."""
        return TaskState(
            x=self.x,
            task_idx=self.task_idx,
            y=self.y,
            value=self.value,
            depth=self.depth,
        )
