package uni

type ObjWrapper struct {
	r *Result
	o Function
}

func (f ObjWrapper) Val(x float64) float64 {
	f.r.FunEvals++
	return f.o.Val(x)
}

type ObjDerivWrapper struct {
	r *Result
	o Deriv
}

func (f ObjDerivWrapper) Val(x float64) float64 {
	f.r.FunEvals++
	return f.o.Val(x)
}

func (f ObjDerivWrapper) ValDeriv(x float64) (float64, float64) {
	f.r.FunEvals++
	f.r.DerivEvals++
	return f.o.ValDeriv(x)
}
