package uni

import "time"

//Solver independent optimization parameters
type Params struct {
	Termination
}

func NewParams() *Params {
	return &Params{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Second,
		},
	}
}
