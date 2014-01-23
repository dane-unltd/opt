package multi

import ()

type fWrapper struct {
	r *Result
	f F
}

func (wrap fWrapper) F(x []float64) float64 {
	wrap.r.FunEvals++
	return wrap.f.F(x)
}

type fdfWrapper struct {
	r   *Result
	fdf FdF
}

func (wrap fdfWrapper) F(x []float64) float64 {
	wrap.r.FunEvals++
	return wrap.fdf.F(x)
}

func (wrap fdfWrapper) DF(x, g []float64) {
	wrap.r.GradEvals++
	wrap.fdf.DF(x, g)
}

func (wrap fdfWrapper) FdF(x, g []float64) float64 {
	wrap.r.FunEvals++
	wrap.r.GradEvals++
	return wrap.fdf.FdF(x, g)
}
