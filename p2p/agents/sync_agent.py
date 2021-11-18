from abc import abstractmethod
from p2p.agents.abstract_agent import *


class SyncAgent(Agent):

    @abstractmethod
    def update_local_parameters(self):
        """
            Method to be called on the end of loop
        :return:
        """
        pass
