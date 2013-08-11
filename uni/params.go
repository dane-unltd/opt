package uni

import "time"

//Solver independent optimization parameters
type Params struct {
	FunTolAbs float64
	FunTolRel float64
	XTolAbs   float64
	XTolRel   float64

	Inexact   bool
	Armijo    float64
	Curvature float64

	IterMax    int
	TimeMax    time.Duration
	FunEvalMax int
}

func NewParams() *Params {
	return &Params{
		FunTolAbs: 1e-15,
		FunTolRel: 1e-15,
		XTolAbs:   1e-6,
		XTolRel:   1e-2,

		Inexact:   true,
		Armijo:    0.2,
		Curvature: 0.9,

		IterMax:    1000,
		TimeMax:    time.Second,
		FunEvalMax: 1000,
	}
}
