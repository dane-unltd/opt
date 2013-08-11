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
	*Solution
	Stats
	Status Status
}

func NewResult(in *Solution) *Result {
	r := &Result{}
	r.Solution = &Solution{}
	r.Solution.SetX(in.X, true)
	r.ObjX = in.ObjX
	if in.GradX != nil {
		r.GradX = make(mat.Vec, len(in.GradX))
		r.GradX.Copy(in.GradX)
	}
	return r
}
