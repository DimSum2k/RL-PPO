##from collections import deque

class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        #self.logprobs = []
        self.rewards = []
        self.dones = []
        #self.values = []
        ##self.probs = deque(maxlen=2)
    
    def clear_memory(self):
        del self.actions[:]
        #del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        #del self.values[:]
        del self.observations[:]
        assert len(self.actions)==len(self.rewards)==len(self.dones)==len(self.observations)==0
        
    def __len__(self):
        #assert len(self.actions)==len(self.logprobs)==len(self.rewards)==len(self.dones)==len(self.values)
        return len(self.actions)