# FLIP-RL
Using Reinforcement learning to determine the optimal placement order of fibres in pick-and-place multi-object spectrographs

## The Problem:

WEAVE has two optimisation problems. The first is how its ~1000 optical fibres get assigned to targets in the field of view; This is handled using a Simulated Annealing algorithm which performs very well. The second is a travelling salesman problem, where the order in which each positioning robot places the optical fibres is decided. This currently uses the program Delta, which uses the nearest available fibre approach. This works reasonably well however, due to both robots having a common shared area in the centre of the field there is a tendency for this algorithm to pause the movement of one robot whilst the other continuously places fibres in the shared area.

My goal is to determine if RL can be successfully applied to these fibre positioning systems, and then study its behaviour as 2 or even more agents explore the environment. At this stage, we are constructing a simplified version which excludes fibre crossing rules to establish a basic version to study its effectiveness. This can then be built up further and adapted to different Multi-Object spectrograph positioning systems.

### To install


### Module components
