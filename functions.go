package opt

import (
	. "github.com/dane-unltd/linalg/matrix"
)

//objective of the form x'*A*x + b'*x + c
func makeQuadratic(A *Dense, b Vec, c float64) Objective {
	m, n := A.Size()
	At := A.TrView()
	if m != n {
		panic("coeff matrix has to be quadratic")
	}
	temp := NewVec(m)
	fun := func(x Vec) float64 {
		val := 0.0
		temp.Mul(A, x)
		val += Dot(x, temp)
		val += Dot(x, b)
		val += c
		return val
	}
	grad := func(x Vec, g Vec) {
		temp.Mul(At, x)
		g.Mul(A, x)
		g.Add(g, temp)
		g.Add(g, b)
	}
	return Objective{fun, grad}
}
