package linprog

import (
	"github.com/dane-unltd/linalg/mat"
)

type Problem struct {
	C, B mat.Vec
	A    *mat.Dense
}

func NewStandard(c mat.Vec, A *mat.Dense, b mat.Vec) *Problem {
	m, n := A.Dims()
	if len(c) != n || len(b) != m {
		panic("linprog: dimension mismatch")
	}
	return &Problem{
		C: c,
		B: b,
		A: A,
	}
}
