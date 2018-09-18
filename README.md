# Starcraft 2 AI with neural networking training

Starcraft 2 AI making in-game decisions on the fly based on previous training through a 2d convolutional neural network.
Picks 1 out of 14 avaliable decisions. Ranging from what to build, when to scout and attack/defense.

First gen: only made decisions on what do to with army units.

Second gen: Picks 1 out of 14 avaliable decisions. Ranging from what to build, when to scout and attack/defense.

## Getting Started

Does not work on python 3.7! Websockets have changed! Use python 3.6.

Download and install Starcraft 2 for free through Blizzards application.

Download maps from [Blizzards github](https://github.com/Blizzard/s2client-proto#map-packs)

Download [python-sc2](https://github.com/Dentosal/python-sc2) by Dentosal

Modify paths.py if you placed Starcraft 2 in non-default place. Give the full path to your sc2 executable under BASEDIR.
paths.py can be found inside of your ~/python36/Lib/site-packages/sc2/paths.py

## Dependencies
##### [python-sc2](https://github.com/Dentosal/python-sc2) by Dentosal

##### [python-sc2 fork](https://github.com/daniel-kukiela/python-sc2) by daniel
adds function on_end which passes game result

##### Keras

##### Tensorflow

##### Opencv

### Other
Python 3.6