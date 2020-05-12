class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        self.observations[:]
        
    def __len__(self):
        assert len(self.actions)==len(self.logprobs)==len(self.rewards)==len(self.dones)==len(self.values)
        return len(self.actions)