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

	IterMax int
	TimeMax time.Duration
}
