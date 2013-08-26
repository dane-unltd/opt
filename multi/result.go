package multi

import (
	"github.com/dane-unltd/linalg/mat"
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
		r.Grad = make(mat.Vec, len(in.Grad))
		r.Grad.Copy(in.Grad)
	}
	return r
}
