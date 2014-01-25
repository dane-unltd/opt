package linprog

import (
	"github.com/gonum/blas/blasw"
)

type Problem struct {
	C, B []float64
	A    goblas.General
}

func NewStandard(c []float64, A General, b []float64) *Problem {
	if len(c) != A.N || len(b) != A.M {
		panic("linprog: dimension mismatch")
	}
	return &Problem{
		C: c,
		B: b,
		A: A,
	}
}
