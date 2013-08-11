package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type ObjWrapper struct {
	r *Result
	o Function
}

func (f ObjWrapper) Val(x mat.Vec) float64 {
	f.r.FunEvals++
	return f.o.Val(x)
}

type ObjGradWrapper struct {
	r *Result
	o Grad
}

func (f ObjGradWrapper) Val(x mat.Vec) float64 {
	f.r.FunEvals++
	return f.o.Val(x)
}

func (f ObjGradWrapper) ValGrad(x, g mat.Vec) float64 {
	f.r.FunEvals++
	f.r.GradEvals++
	return f.o.ValGrad(x, g)
}
