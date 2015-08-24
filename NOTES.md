# Notes
===
Tidbits I learned while making this:

* The following items improved network performance
	* Multiplying randomly initialized weights by 0.01
	* Calibrating randomly initialized weights by factor described at [here](http://cs231n.github.io/neural-networks-2/#reg). (Not sure if increased, seems like it though)
	* Scaling inputs to [-1,1] span. This one helped a lot
	* *Not* scaling targets to [-1,1]. This helped a lot too
	* Adding regularization
	* Momentum works fairly well. More tests need to be run to see how well
	
* The following items reduce or didn't affect network performance
	* Validation failure happens very rarely. May not be necessary
	
* Miscellaneous Notes:
	* Start with a low number of hidden nodes and slowly increase to (hopefully) increase accuracy	