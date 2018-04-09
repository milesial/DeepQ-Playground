class EpsilonDecay(object):
    """
    Class that helps getting the epsilon that controls exploration.
    The value get from 'start_value' to 'end_value' in 'duration' steps, linearly.
    """

    def __init__(self, start_value, end_value, duration):
        super(EpsilonDecay, self).__init__()
        self.value = start_value
        self.end_value = end_value
        self.decrement = (start_value - end_value) / duration

    def get(self):
        old_value = self.value
        if self.value > self.end_value:
            self.value -= self.decrement

        return old_value
