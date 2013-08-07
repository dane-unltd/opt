package linprog

import "time"

//Solver independent optimization parameters
type Params struct {
	Infeasibility float64
	DualityGap    float64

	IterMax int
	TimeMax time.Duration
}
