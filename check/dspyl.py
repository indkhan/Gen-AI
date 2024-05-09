import sys

import os
import dspy

from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate
from rich import print


turbo = ChatOpenAI(model="gpt-3.5")
