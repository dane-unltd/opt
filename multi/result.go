package multi

import (
	"time"
)

type Stats struct {
	Iter      int
	Time      time.Duration
	FunEvals  int
	GradEvals int
}

type Result struct {
	Solution
	Stats
	Status Status
}

func NewResult(in *Solution) *Result {
	r := &Result{}
	r.Solution.SetX(in.X, true)
	r.Obj = in.Obj
	if in.Grad != nil {
		r.Grad = make([]float64, len(in.Grad))
		copy(r.Grad, in.Grad)
	}
	return r
}
