package linprog

import (
	"github.com/dane-unltd/linalg/mat"
	"time"
)

type Stats struct {
	Iter int
	Time time.Duration
}

type Result struct {
	Solution
	Rp, Rd, Rs mat.Vec
	Stats
	Status Status
}

func NewResult(prob *Problem) *Result {
	r := &Result{}
	m, n := prob.A.Dims()
	r.X = mat.NewVec(n)
	r.S = mat.NewVec(n)
	r.Y = mat.NewVec(m)

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
