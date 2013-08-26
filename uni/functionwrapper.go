package uni

type fWrapper struct {
	r *Result
	f F
}

func (wrap fWrapper) F(x float64) float64 {
	wrap.r.FunEvals++
	return wrap.f.F(x)
}

type fdfWrapper struct {
	r   *Result
	fdf FdF
}

func (wrap fdfWrapper) F(x float64) float64 {
	wrap.r.FunEvals++
	return wrap.fdf.F(x)
}

func (wrap fdfWrapper) DF(x float64) float64 {
	wrap.r.DerivEvals++
	return wrap.fdf.DF(x)
}

func (wrap fdfWrapper) FdF(x float64) (float64, float64) {
	wrap.r.FunEvals++
	wrap.r.DerivEvals++
	return wrap.fdf.FdF(x)
}
