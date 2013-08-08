package uni

type Function interface {
	Val(float64) float64
}

type Deriv interface {
	Function
	ValDeriv(float64) (float64, float64)
}

type Deriv2 interface {
	Deriv
	ValDeriv2(float64) (float64, float64, float64)
}
