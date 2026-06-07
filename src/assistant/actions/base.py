"""Abstract Action base class: every action type implements execute() and describe() — implemented in task 1.6.1."""
from abc import ABC, abstractmethod


class Action(ABC):
    @abstractmethod
    def execute(self, details: dict) -> str:
        pass

    @abstractmethod
    def describe(self, details: dict) -> str:
        pass
