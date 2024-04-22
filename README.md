# IRDropDG
Using gradient descent method to solve the IR-Drop problem in large-scale display panel with PMOS driving transistor. GD method is realized with Pytorch.

## IR-Drop problem in displays
Display panel can be modeled as a discrete gird system with parasitic resistence and non-linear voltage-controlled current source:
![IR-Drop](https://github.com/YunTing-k/IRDropGD/blob/master/img/img1.png?raw=true)
Points can be classified into tree types:
![Grid Type](https://github.com/YunTing-k/IRDropGD/blob/master/img/img2.png?raw=true)
Inner point's grid current can be derived by the four nearby grid's voltage:
![func1](https://github.com/YunTing-k/IRDropGD/blob/master/img/img3.png?raw=true)
where $𝐺_(𝑖,𝑗)=𝑔_((𝑖,𝑗)(𝑖−1,𝑗))+ 𝑔_((𝑖,𝑗)(𝑖,𝑗−1))+ 𝑔_((𝑖,𝑗)(𝑖,𝑗+1))+ 𝑔_((𝑖,𝑗)(𝑖+1,𝑗))$ is the sum of conductance of nearby points, $𝑔_((𝑖,𝑗)(𝑖−1,𝑗))$ is the conductance between grid(i, j) and grid(i-1, j).
Pixel's current can be derived by $𝑉_(𝑖,𝑗)$:
![func2](https://github.com/YunTing-k/IRDropGD/blob/master/img/img4.png?raw=true)
The whole system need to reach a steady state with the grid current equals the pixel current. Therefore, one equation can be derived:
![func3](https://github.com/YunTing-k/IRDropGD/blob/master/img/img5.png?raw=true)
Similarly, for four corner points we have:
![func4](https://github.com/YunTing-k/IRDropGD/blob/master/img/img6.png?raw=true)
for four points on four edges, we have:
![func5](https://github.com/YunTing-k/IRDropGD/blob/master/img/img7.png?raw=true)
With M-row, N-col panels, MN equation can be derived. To apply the GD method, loss is defined as follow:
![func6](https://github.com/YunTing-k/IRDropGD/blob/master/img/img8.png?raw=true)
Grad can be derived by auotgrad in pytorch:
![func7](https://github.com/YunTing-k/IRDropGD/blob/master/img/img9.png?raw=true)

## Boundary condition fixation
Some of grids are assigned as the current injection points, they must have a stable voltage. Thus, we need to eliminate the gradient of corresponding points by boundary condition fixation.
To realize, these points' $𝐼_(𝑖,𝑗)$ and $𝑓(𝑉_(𝑖,𝑗))$ are set to zero (since the equation is not satisfied at these points due to the current injection) and the referred voltage used in nearby points' $𝐼_(𝑖,𝑗)$ is set as a constant number with **detach()** method in Pytorch.

## Architecture
![Architecture](https://github.com/YunTing-k/IRDropGD/blob/master/img/img10.png?raw=true)
## Customizable parameters of IR-Drop solver
![Customizable parameters](https://github.com/YunTing-k/IRDropGD/blob/master/img/img11.png?raw=true)

## Note
This project's accuracy is bad when the scale of panel is over **[100*100]**. Meawhile, it shows an unstable convergence behavior under different initialization. It is just a small demo for reasearch and doesn't guarantee the accuracy!
In author's opinion, the bad solution quality may comes from the lack of well-defined constraints of the current. Moreover, gradient descent may sholud't have the same weight during the update (at least during the first iterations).
