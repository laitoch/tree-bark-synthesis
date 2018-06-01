- A Pipeline for Synthesizing Tree Bark Textures from a Single Input Image -

ABOUT:

This CD is an attachment for a Bachelor Thesis.

Thesis type: Bachelor Thesis
Author: Petr Laitoch
Title: Procedural Modeling of Tree Bark
Year: 2018
Submission date: May 18, 2018
University: Charles University
Faculty: Faculty of Mathematics and Physics
Location: Prague

INSTALL:

To install our implementation of a bark generation pipeline, check that you are
running Linux and have installed the following software on your distribution
• C++ compiler (gcc)
• Python 2
• pip
and run the script setup.sh.

RUN:

• Go into any subdirectory of ./experiments/ containing a makefile and run.py.

We recommend choosing an experiment with lower hardware requirements and a
faster runtime. At least at first. For example
experiments/current1/eucalyptus_2/ is a good choice.

• Run make.
• It may run a long time. (Tens of minutes)
• You should see intermediate results in subdirectories soon, though.

RENDER:

Run vp.py scripts in the subdirectories of display_3d.
