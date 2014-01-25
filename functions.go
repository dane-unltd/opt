package opt

import (
	"github.com/dane-unltd/goblas"
	"github.com/gonum/blas"
	"math"
)

//objective of the form x'*A*x + b'*x + c
type Quadratic struct {
	A goblas.General
	B goblas.Vector
	C float64

	temp goblas.Vector
}

func NewQuadratic(A goblas.General, b []float64, c float64) *Quadratic {
	if A.M != A.N {
		panic("matrix has to be quadratic")
	}
	if A.N != len(b) {
		panic("dimension mismatch between A and b")
	}
	return &Quadratic{
		A:    A,
		B:    goblas.NewVector(b),
		C:    c,
		temp: goblas.NewVector(make([]float64, A.N)),
	}
}

func (Q *Quadratic) F(xs []float64) float64 {
	x := goblas.NewVector(xs)
	goblas.Dcopy(Q.B, Q.temp)
	goblas.Dgemv(blas.NoTrans, 1, Q.A, x, 1, Q.temp)
	return goblas.Ddot(x, Q.temp) + Q.C
}

func (Q *Quadratic) DF(xs, gs []float64) {
	x := goblas.NewVector(xs)
	g := goblas.NewVector(gs)

	goblas.Dcopy(Q.B, g)
	goblas.Dgemv(blas.NoTrans, 1, Q.A, x, 1, g)
	goblas.Dgemv(blas.Trans, 1, Q.A, x, 1, g)
}

func (Q *Quadratic) FdF(xs, gs []float64) float64 {
	x := goblas.NewVector(xs)
	g := goblas.NewVector(gs)

	goblas.Dcopy(Q.B, g)
	goblas.Dgemv(blas.NoTrans, 1, Q.A, x, 1, g)

	val := goblas.Ddot(g, x) + Q.C

	goblas.Dgemv(blas.Trans, 1, Q.A, x, 1, g)

	return val
}

type Rosenbrock struct{}

func (R Rosenbrock) F(x []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x)-1; i++ {
		sum += math.Pow(1-x[i], 2) +
			100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
	}
	return sum
}

func (R Rosenbrock) DF(x, g []float64) {
	g[len(x)-1] = 0
	for i := 0; i < len(x)-1; i++ {
		g[i] = -1 * 2 * (1 - x[i])
		g[i] += 2 * 100 * (x[i+1] - math.Pow(x[i], 2)) * (-2 * x[i])
	}
	for i := 1; i < len(x); i++ {
		g[i] += 2 * 100 * (x[i] - math.Pow(x[i-1], 2))
	}
}

func (R Rosenbrock) FdF(x, g []float64) float64 {
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

func (R RealPlus) Project(x []float64) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}
