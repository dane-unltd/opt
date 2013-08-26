package linprog

import "time"

//Solver independent optimization parameters
type Params struct {
	Infeasibility float64
	DualityGap    float64

	Termination
}

func NewParams() *Params {
	return &Params{
		Infeasibility: 1e-8,
		DualityGap:    1e-8,
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
	}
}
