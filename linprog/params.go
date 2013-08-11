package linprog

import "time"

//Solver independent optimization parameters
type Params struct {
	Infeasibility float64
	DualityGap    float64

	IterMax int
	TimeMax time.Duration
}

func NewParams() *Params {
	return &Params{
		Infeasibility: 1e-8,
		DualityGap:    1e-8,
		IterMax:       1000,
		TimeMax:       time.Minute,
	}
}
