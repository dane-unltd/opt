package multi

import "time"

//Solver independent optimization parameters
type Params struct {
	Termination

	Accuracy float64

	FunEvalMax int
}

func NewParams() *Params {
	return &Params{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
		Accuracy:   1e-6,
		FunEvalMax: 10000,
	}
}
