import unittest
from types import SimpleNamespace
from unittest.mock import patch

from liveness import Challenge


class ChallengeTests(unittest.TestCase):
    @staticmethod
    def landmarks(nose_x):
        lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(455)]
        lm[234] = SimpleNamespace(x=0.2, y=0.5)
        lm[454] = SimpleNamespace(x=0.8, y=0.5)
        lm[1] = SimpleNamespace(x=nose_x, y=0.5)
        return lm

    def test_challenge_requires_turn_then_front(self):
        with patch("liveness.random.choice", return_value="turn_left"):
            challenge = Challenge()

        self.assertEqual(challenge.steps, ["turn_left", "look_front"])

        status = "running"
        for _ in range(3):
            status = challenge.update(self.landmarks(0.30), (480, 640, 3), 0.9)
        self.assertEqual(status, "running")
        self.assertEqual(challenge.current, "look_front")

        for _ in range(3):
            status = challenge.update(self.landmarks(0.50), (480, 640, 3), 0.9)
        self.assertEqual(status, "passed")
