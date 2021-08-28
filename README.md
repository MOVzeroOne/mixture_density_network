# mixture density network
multimodal distributions with neural networks


<br> <b> Circle </b> <br>
The goal is to make a network that maps x to y where for every x there can be multiple y's, as in this case x,y forms a circle. <br>
This is done by learning multiple normalized gaussian's and combining them into a joint probability distribution. <br>
Sampling from this distribution will then give the model the ability to represent multiple values for one input. <br>

![](circle.gif)

<br>

<br> <b> bimodal distribution </b> <br>
Goal is to directly learn a bimodal distribution for a fixed input. <br>
This is done by first predefining a fixed input, and then sampling target points around 5 and -5. <br>

![](bimodal_density.gif)  
 