package uni

import "time"

//Solver independent optimization parameters
type Params struct {
	Accuracy float64

	Armijo    float64
	Curvature float64

	IterMax    int
	TimeMax    time.Duration
	FunEvalMax int
}

func NewParams() *Params {
	return &Params{
		Accuracy: 1e-3,

		Armijo:    0.2,
		Curvature: 0.9,

		IterMax:    1000,
		TimeMax:    time.Second,
		FunEvalMax: 10000,
	}
}

func addConvChecks(upd *[]Updater, p *Params, r *Result) {
	(*upd) = append(*upd, Accuracy(p.Accuracy))
	(*upd) = append(*upd, IterMax(p.IterMax))
	(*upd) = append(*upd, TimeMax(p.TimeMax))
	if p.Curvature > 0 {
		w := Wolfe{
			Armijo:    p.Armijo,
			Curvature: p.Curvature,
			X0:        r.XLower,
			F0:        r.ObjLower,
			Deriv0:    r.DerivLower,
		}
		(*upd) = append(*upd, w)
	}
}
