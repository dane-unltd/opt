package multi

import "time"

//Solver independent optimization parameters
type Params struct {
	FunTolAbs float64
	FunTolRel float64
	XTolAbs   float64
	XTolRel   float64

	IterMax int
	TimeMax time.Duration
}

func NewParams() *Params {
	return &Params{
		FunTolAbs: 1e-15,
		FunTolRel: 1e-15,
		XTolAbs:   1e-6,
		XTolRel:   1e-2,

		IterMax: 1000,
		TimeMax: time.Second,
	}
}
