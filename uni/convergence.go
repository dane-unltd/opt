package uni

import (
	"math"
)

type Accuracy float64

func (acc Accuracy) Update(r *Result) Status {
	if math.Abs(r.XUpper-r.XLower) < float64(acc) {
		return XAbsConv
	}
	return NotTerminated
}

type Wolfe struct {
	Armijo         float64
	Curvature      float64
	X0, F0, Deriv0 float64
}

func (w Wolfe) Update(r *Result) Status {
	if math.Abs(r.Deriv/w.Deriv0) < w.Curvature &&
		r.Obj-w.F0 < w.Armijo*(r.X-w.X0)*w.Deriv0 {
		return WolfeConv
	}
	return NotTerminated
}
