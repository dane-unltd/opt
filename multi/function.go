package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type Function interface {
	Val(x mat.Vec) float64
}

type Grad interface {
	Function
	ValGrad(x, g mat.Vec) float64
}

type Hessian interface {
	Grad
	ValGradHess(x, g, h mat.Vec) float64
}

type Projection interface {
	Project(x mat.Vec)
}
