from user_simulator import *

class Journey:

    def __init__(self, user: User):
        self._reset_next_step = True
        self.user = user
        self.styles = ["style 1", "style 2", "style 3"]
        self.page = 0
        self.current_style = 0

    def step(self, action):
        if self._reset_next_step:
            return self.reset()
        self.current_style = action
        cont = self.user.interact(self.page, self.styles[action])
        print(f'Page: {self.page} Style: {self.current_style}')
        if cont:
            self.page +=1
            if self.page == 5:
                self._reset_next_step = True
                return self.observe(), 10, True
            else:
                return self.observe(), 1, False
        else:
            self._reset_next_step = True
            return self.observe(), -5, True

    def observe(self):
        return (self.current_style, self.page-1, self.user.user_type)

    def reset(self):
        self.page = 0
        self.current_style = 0
        self._reset_next_step = False
        return self.observe(), None, False