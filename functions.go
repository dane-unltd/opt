package opt

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

//objective of the form x'*A*x + b'*x + c
type Quadratic struct {
	A *mat.Dense
	B mat.Vec
	C float64

	temp mat.Vec
}

func NewQuadratic(A *mat.Dense, b mat.Vec, c float64) *Quadratic {
	m, n := A.Dims()
	if m != n {
		panic("matrix has to be quadratic")
	}
	if n != len(b) {
		panic("dimension mismatch between A and b")
	}
	return &Quadratic{
		A:    A,
		B:    b,
		C:    c,
		temp: mat.NewVec(n),
	}
}

func (Q *Quadratic) Val(x mat.Vec) float64 {
	val := 0.0
	Q.temp.Transform(Q.A, x)
	val += mat.Dot(x, Q.temp)
	val += mat.Dot(x, Q.B)
	val += Q.C
	return val
}

func (Q *Quadratic) ValGrad(x, g mat.Vec) float64 {
	At := Q.A.TrView()

	Q.temp.Transform(Q.A, x)

	g.Transform(At, x)
	g.Add(g, Q.temp)
	g.Add(g, Q.B)

	val := 0.0
	val += mat.Dot(x, Q.temp)
	val += mat.Dot(x, Q.B)
	val += Q.C
	return val
}

type Rosenbrock struct{}

func (R Rosenbrock) Val(x mat.Vec) float64 {
	sum := 0.0
	for i := 0; i < len(x)-1; i++ {
		sum += math.Pow(1-x[i], 2) +
			100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
	}
	return sum
}

func (R Rosenbrock) ValGrad(x, g mat.Vec) float64 {
	g[len(x)-1] = 0
	for i := 0; i < len(x)-1; i++ {
		g[i] = -1 * 2 * (1 - x[i])
		g[i] += 2 * 100 * (x[i+1] - math.Pow(x[i], 2)) * (-2 * x[i])
	}
	for i := 1; i < len(x); i++ {
		g[i] += 2 * 100 * (x[i] - math.Pow(x[i-1], 2))
	}

	sum := 0.0
	for i := 0; i < len(x)-1; i++ {
		sum += math.Pow(1-x[i], 2) +
			100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
	}
	return sum
}

type RealPlus struct{}

func (R RealPlus) Project(x mat.Vec) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}
