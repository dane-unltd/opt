package linprog

import (
	"github.com/dane-unltd/linalg/clapack"
	"github.com/dane-unltd/linalg/mat"
	"github.com/kortschak/cblas"
	"testing"
)

type matops struct {
	cblas.Blas
	clapack.Lapack
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

	mdl := NewStandard(c, A, b)

	sol := NewPredCorr()
	sol.Solve(mdl)

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	rd.Sub(c, mdl.S)
	rd.AddMul(At, mdl.Y, -1)
	rp.Apply(A, mdl.X)
	rp.Sub(b, rp)
	rs.Mul(mdl.X, mdl.S)
	rs.Neg(rs)
	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if dev > tol {
		t.Fail()
	}

	temp := mat.NewVec(n)
	temp.Sub(mdl.X, xStar)

	if temp.Nrm2() > tol {
		t.Log(mdl.X)
		t.Fail()
	}
}

func TestLinprog(t *testing.T) {
	m := 500
	n := 1000
	tol := 1e-8

	A := mat.RandN(m, n)
	c := mat.RandVec(n)
	b := mat.NewVec(m)
	xt := mat.RandVec(n)
	b.Apply(A, xt)

	At := A.TrView()

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	mdl := NewStandard(c, A, b)

	//Example for printing duality gap and infeasibilities
	mdl.AddCallback(NewDisplay(2).Update)

	sol := NewPredCorr()
	sol.Solve(mdl)

	rd.Sub(c, mdl.S)
	rd.AddMul(At, mdl.Y, -1)
	rp.Apply(A, mdl.X)
	rp.Sub(b, rp)
	rs.Mul(mdl.X, mdl.S)
	rs.Neg(rs)

	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if dev > tol {
		t.Log(dev)
		t.Fail()
	}
}

func BenchmarkLinprog(bench *testing.B) {
	bench.StopTimer()
	m := 50
	n := 100
	tol := 1e-3
	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	for i := 0; i < bench.N; i++ {
		A := mat.RandN(m, n)
		c := mat.RandVec(n)
		b := mat.NewVec(m)
		xt := mat.RandVec(n)
		b.Apply(A, xt)

		At := A.TrView()

		mdl := NewStandard(c, A, b)
		sol := NewPredCorr()
		bench.StartTimer()
		sol.Solve(mdl)
		bench.StopTimer()

		rd.Sub(c, mdl.S)
		rd.AddMul(At, mdl.Y, -1)
		rp.Apply(A, mdl.X)
		rp.Sub(b, rp)
		rs.Mul(mdl.X, mdl.S)
		rs.Neg(rs)

		dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
		if dev > tol {
			bench.Log(dev)
		}
	}
}
