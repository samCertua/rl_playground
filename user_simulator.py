import random


class User:

    def __init__(self, correct_sequence, user_type):
        self.correct_sequence = correct_sequence
        self.user_type = user_type

    def interact(self, page: int, style: str):
        if self.correct_sequence[page] == style:
            return True
        else:
            return False


class OldMan(User):

    def __init__(self):
        correct_sequence = ["style 1", "style 2", "style 3", "style 1", "style 2"]
        super(OldMan, self).__init__(correct_sequence, 0)


class YoungWoman(User):

    def __init__(self):
        correct_sequence = ["style 1", "style 2", "style 2", "style 3", "style 1"]
        super(YoungWoman, self).__init__(correct_sequence, 1)

class InconsistentMan(User):

    def __init__(self):
        correct_sequence = ["style 1", "style 1", "style 1", "style 3", "style 1"]
        super(InconsistentMan, self).__init__(correct_sequence, -1)

    def interact(self, page: int, style: str):
        if self.correct_sequence[page] == style:
            return random.choices(population=[True, False], weights=[0.9,0.1],k=1)[0]
        else:
            return random.choices(population=[True, False], weights=[0.1,0.9],k=1)[0]
