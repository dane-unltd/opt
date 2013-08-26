package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type fWrapper struct {
	r *Result
	f F
}

func (wrap fWrapper) F(x mat.Vec) float64 {
	wrap.r.FunEvals++
	return wrap.f.F(x)
}

type fdfWrapper struct {
	r   *Result
	fdf FdF
}

func (wrap fdfWrapper) F(x mat.Vec) float64 {
	wrap.r.FunEvals++
	return wrap.fdf.F(x)
}

func (wrap fdfWrapper) DF(x, g mat.Vec) {
	wrap.r.GradEvals++
	wrap.fdf.DF(x, g)
}

func (wrap fdfWrapper) FdF(x, g mat.Vec) float64 {
	wrap.r.FunEvals++
	wrap.r.GradEvals++
	return wrap.fdf.FdF(x, g)
}
