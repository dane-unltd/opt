package opt

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/dbw"
	"math"
)

//objective of the form x'*A*x + b'*x + c
type Quadratic struct {
	A dbw.General
	B dbw.Vector
	C float64

	temp dbw.Vector
}

func NewQuadratic(A dbw.General, b []float64, c float64) *Quadratic {
	if A.Rows != A.Cols {
		panic("matrix has to be quadratic")
	}
	if A.Cols != len(b) {
		panic("dimension mismatch between A and b")
	}
	return &Quadratic{
		A:    A,
		B:    dbw.NewVector(b),
		C:    c,
		temp: dbw.NewVector(make([]float64, A.Cols)),
	}
}

func (Q *Quadratic) F(xs []float64) float64 {
	x := dbw.NewVector(xs)
	dbw.Copy(Q.B, Q.temp)
	dbw.Gemv(blas.NoTrans, 1, Q.A, x, 1, Q.temp)
	return dbw.Dot(x, Q.temp) + Q.C
}

func (Q *Quadratic) DF(xs, gs []float64) {
	x := dbw.NewVector(xs)
	g := dbw.NewVector(gs)

	dbw.Copy(Q.B, g)
	dbw.Gemv(blas.NoTrans, 1, Q.A, x, 1, g)
	dbw.Gemv(blas.Trans, 1, Q.A, x, 1, g)
}

func (Q *Quadratic) FdF(xs, gs []float64) float64 {
	x := dbw.NewVector(xs)
	g := dbw.NewVector(gs)

	dbw.Copy(Q.B, g)
	dbw.Gemv(blas.NoTrans, 1, Q.A, x, 1, g)

	val := dbw.Dot(g, x) + Q.C

	dbw.Gemv(blas.Trans, 1, Q.A, x, 1, g)

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
