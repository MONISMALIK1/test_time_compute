"""A tiny set of grade-school math word problems with known answers.

GSM8K-style (single numeric answer reachable by a few arithmetic steps), but
hand-authored and original so nothing here leaks from a public test set. Each
problem has a stable ``id`` — the compute-optimal allocator in ``allocate.py``
keys its per-question budget on it — and a gold ``answer`` string graded
numerically by ``extract.grade``.

These are the problems ``python -m test_time_compute --bench`` runs to show the
accuracy curve as you add test-time compute.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str


SAMPLE_PROBLEMS: list[Problem] = [
    Problem("p01", "A baker makes 24 muffins and packs them into boxes of 6. "
                   "How many boxes does she fill?", "4"),
    Problem("p02", "Tom has 15 marbles. He gives 4 to his sister and then buys 9 more. "
                   "How many marbles does he have now?", "20"),
    Problem("p03", "A train travels at 60 miles per hour for 3 hours. "
                   "How many miles does it travel?", "180"),
    Problem("p04", "A shirt costs $20 and is on sale for 25% off. "
                   "What is the sale price in dollars?", "15"),
    Problem("p05", "Sarah reads 12 pages each day. "
                   "How many pages does she read in 2 weeks?", "168"),
    Problem("p06", "A rectangle is 8 cm long and 5 cm wide. "
                   "What is its area in square centimeters?", "40"),
    Problem("p07", "There are 32 students in a class. If they form teams of 4, "
                   "how many teams are there?", "8"),
    Problem("p08", "A water tank holds 200 liters and is 3/4 full. "
                   "How many liters of water are in it?", "150"),
    Problem("p09", "John buys 3 notebooks at $4 each and 2 pens at $2 each. "
                   "How much does he spend in dollars?", "16"),
    Problem("p10", "A farmer has 5 hens and each hen lays 2 eggs per day. "
                   "How many eggs are collected in 6 days?", "60"),
    Problem("p11", "A pizza is cut into 8 slices. Three friends each eat 2 slices. "
                   "How many slices are left?", "2"),
    Problem("p12", "A book has 240 pages and Maria has read 1/3 of it. "
                   "How many pages are left to read?", "160"),

    # Harder, multi-step — where the test-time-compute advantage shows up, because
    # a single greedy pass slips more often than a voted-over-many one.
    Problem("p13", "A store sells pencils at 3 for $2. "
                   "How much do 12 pencils cost in dollars?", "8"),
    Problem("p14", "A tank fills at 4 liters per minute. "
                   "How many liters are in it after 1 hour and 15 minutes?", "300"),
    Problem("p15", "Maria buys 4 books at $7 each and gets a $5 discount at checkout. "
                   "How much does she pay in dollars?", "23"),
    Problem("p16", "A car travels 150 km in 2 hours, then 90 km in the next hour. "
                   "What is its average speed for the whole trip, in km/h?", "80"),
    Problem("p17", "In a class of 30 students, 40% play soccer. "
                   "How many students do not play soccer?", "18"),
    Problem("p18", "Tom earns $15 per hour and works 8 hours a day for 5 days, "
                   "then spends $120. How much money does he have left in dollars?", "480"),
]
