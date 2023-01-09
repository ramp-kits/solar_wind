# RAMP starting kit on solar wind classification


_Authors: Gautier Nguyen, Joris van den Bossche, Nicolas Aunai & Balazs Kegl_

Interplanetary Coronal Mass Ejections (ICMEs) result from magnetic instabilities occurring in the Sun atmosphere, and interact with the planetary environment and may result in intense internal activity such as strong particle acceleration, so-called geomagnetic storms and geomagnetic induced currents. These effects have serious consequences regarding space and ground technologies and understanding them is part of the so-called space weather discipline.

ICMEs signatures as measured by in-situ spacecraft come as patterns in time series of the magnetic field, the particle density, bulk velocity, temperature etc. Although well visible by expert eyes, these patterns have quite variable characteristics which make naive automatization of their detection difficult.

The goal of this RAMP is to detect Interplanetary Coronal Mass Ejections (ICMEs) in the data measured by in-situ spacecraft.


#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install ramp-workflow
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](solar_wind_starting_kit.ipynb).

To test the starting-kit, run


```
ramp-test --quick-test
```


#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.



