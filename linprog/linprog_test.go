package linprog

import (
	"github.com/dane-unltd/linalg/lapack/lapacke"
	"github.com/dane-unltd/linalg/mat"
	"github.com/gonum/blas/cblas"
	"testing"
)

type matops struct {
	cblas.Blas
	lapacke.Lapack
}

func init() {
	mat.Register(matops{})
}

func TestLinprog2(t *testing.T) {
	m := 1
	n := 5
	tol := 1e-8

	a := mat.NewVec(n).AddSc(1)
	xStar := mat.NewVec(n)
	xStar[0] = 1

	A := mat.NewFromArray(a, true, m, n)
	At := A.TrView()
	b := mat.NewVec(m).AddSc(1)
	c := mat.NewVec(n)
	c[0] = -1

	prob := NewStandard(c, A, b)

	result := Solve(prob, nil)

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	rd.Sub(c, result.S)
	rd.AddMul(At, result.Y, -1)
	rp.Transform(A, result.X)
	rp.Sub(b, rp)
	rs.Mul(result.X, result.S)
	rs.Neg(rs)
	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if dev > tol {
		t.Fail()
	}

	temp := mat.NewVec(n)
	temp.Sub(result.X, xStar)

	if temp.Nrm2() > tol {
		t.Log(result.X)
		t.Fail()
	}
}

func TestLinprog(t *testing.T) {
	m := 1000
	n := 3000
	tol := 1e-8

	A := mat.RandN(m, n)
	c := mat.RandVec(n)
	b := mat.RandVec(m)

	//xt := mat.RandVec(n)
	//b.Transform(A, xt)

	At := A.TrView()

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	prob := NewStandard(c, A, b)
	params := NewParams()
	params.IterMax = 100

	//Example for printing duality gap and infeasibilities
	result := Solve(prob, params, NewDisplay(1))

	rd.Sub(c, result.S)
	rd.AddMul(At, result.Y, -1)
	rp.Transform(A, result.X)
	rp.Sub(b, rp)
	rs.Mul(result.X, result.S)
	rs.Neg(rs)

	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if !(dev < tol) {
		t.Log(dev, result.Status)
		t.Fail()
	}
}

func BenchmarkLinprog(bench *testing.B) {
	bench.StopTimer()
	m := 100
	n := 200
	tol := 1e-3
	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	for i := 0; i < bench.N; i++ {
		A := mat.RandN(m, n)
		c := mat.RandVec(n)
		b := mat.NewVec(m)
		xt := mat.RandVec(n)
		b.Transform(A, xt)

		At := A.TrView()

		prob := NewStandard(c, A, b)
		bench.StartTimer()
		result := Solve(prob, nil)
		bench.StopTimer()

		rd.Sub(c, result.S)
		rd.AddMul(At, result.Y, -1)
		rp.Transform(A, result.X)
		rp.Sub(b, rp)
		rs.Mul(result.X, result.S)
		rs.Neg(rs)

		dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
		if dev > tol {
			bench.Log(dev)
		}
	}
}
