# Project: Weakly supervised learning-- label noise and correction

### Installation
To install all the required libraries, run "pip install -r requirements.txt" in a shell.


### [Full Project Description](doc/project3_desc.md)
Term: Fall 2021

+ Team (Group 1)
+ Team members
	+ Caitlyn Chen
	+ Nikhil Cherukupalli
	+ Yue Li
	+ Bohao Ma
	+ Yarong Wang
	+ Ziyong Zhang

+ Project summary: In this project, for Model I, we create a Convolutional Neural Network (CNN) model, which improves significantly in terms of prediction accuracy compared to the baseline logistic regression model. For model II, we first train a neural network that maps the noisy labels to clean labels, conditional on the input image. Then we use this trained network to get predicted labels from the noisy labels and utilize the predicted labels to train our CNN model in Model I.
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md))
+ Nikhil, Yue, Bohao, Yarong, Ziyong contribute to the building and testing of Model I.
+ Nikhil, Yue, Bohao, Yarong, Ziyong contribute to the building and testing of Model II.
+ All team members approve our work presented in this GitHub repository including this contribution statement.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
