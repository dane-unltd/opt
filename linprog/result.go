package linprog

import (
	"time"
)

type Stats struct {
	Iter int
	Time time.Duration
}

type Result struct {
	Solution
	Rp, Rd, Rs []float64
	Stats
	Status Status
}

func NewResult(prob *Problem) *Result {
	r := &Result{}
	r.X = make([]float64, len(prob.C))
	r.S = make([]float64, len(prob.C))
	r.Y = make([]float64, len(prob.B))

	return r

	/*
		if in.X != nil {
			r.X.Copy(in.X)
		}
		if in.S != nil {
			r.S.Copy(in.S)
		}
		if in.Y != nil {
			r.Y.Copy(in.Y)
		}
	*/
}
