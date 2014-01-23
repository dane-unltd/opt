package opt

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blasw"
	"math"
)

//objective of the form x'*A*x + b'*x + c
type Quadratic struct {
	A blasw.Ge
	B blasw.Vector
	C float64

	temp blasw.Vector
}

func NewQuadratic(A blasw.Ge, b []float64, c float64) *Quadratic {
	if A.M != A.N {
		panic("matrix has to be quadratic")
	}
	if A.N != len(b) {
		panic("dimension mismatch between A and b")
	}
	return &Quadratic{
		A:    A,
		B:    blasw.NewVector(b),
		C:    c,
		temp: blasw.NewVector(make([]float64, A.N)),
	}
}

func (Q *Quadratic) F(xs []float64) float64 {
	x := blasw.NewVector(xs)
	blasw.Copy(Q.B, Q.temp)
	blasw.Gemv(blas.NoTrans, 1, Q.A, x, 1, Q.temp)
	return blasw.Dot(x, Q.temp) + Q.C
}

func (Q *Quadratic) DF(xs, gs []float64) {
	x := blasw.NewVector(xs)
	g := blasw.NewVector(gs)

	blasw.Copy(Q.B, g)
	blasw.Gemv(blas.NoTrans, 1, Q.A, x, 1, g)
	blasw.Gemv(blas.Trans, 1, Q.A, x, 1, g)
}

func (Q *Quadratic) FdF(xs, gs []float64) float64 {
	x := blasw.NewVector(xs)
	g := blasw.NewVector(gs)

	blasw.Copy(Q.B, g)
	blasw.Gemv(blas.NoTrans, 1, Q.A, x, 1, g)

	val := blasw.Dot(g, x) + Q.C

	blasw.Gemv(blas.Trans, 1, Q.A, x, 1, g)

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
