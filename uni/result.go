package uni

import (
	"time"
)

type Stats struct {
	Iter        int
	Time        time.Duration
	FunEvals    int
	DerivEvals  int
	Deriv2Evals int
}

type Result struct {
	Solution
	Stats
	Status Status
}

func NewResult(in *Solution) *Result {
	r := &Result{}
	r.Solution = *in
	return r
}
