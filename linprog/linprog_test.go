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

	mdl := &Model{}

	mdl.A = mat.NewFromArray(a, true, m, n)
	At := mdl.A.TrView()
	mdl.B = mat.NewVec(m)
	mdl.B.AddSc(1)
	mdl.C = mat.NewVec(n)
	mdl.C[0] = -1

	sol := NewPredCorr()
	sol.Solve(mdl)

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	rd.Sub(mdl.C, mdl.S)
	rd.AddMul(At, mdl.Y, -1)
	rp.Apply(mdl.A, mdl.X)
	rp.Sub(mdl.B, rp)
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
	m := 5
	n := 10
	tol := 1e-8

	mdl := &Model{}

	mdl.A = mat.RandN(m, n)
	mdl.C = mat.RandVec(n)
	mdl.B = mat.NewVec(m)
	xt := mat.RandVec(n)
	mdl.B.Apply(mdl.A, xt)

	At := mdl.A.TrView()

	sol := NewPredCorr()
	sol.Solve(mdl)

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	rd.Sub(mdl.C, mdl.S)
	rd.AddMul(At, mdl.Y, -1)
	rp.Apply(mdl.A, mdl.X)
	rp.Sub(mdl.B, rp)
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
		mdl := &Model{}

		mdl.A = mat.RandN(m, n)
		mdl.C = mat.RandVec(n)
		mdl.B = mat.NewVec(m)
		xt := mat.RandVec(n)
		mdl.B.Apply(mdl.A, xt)

		At := mdl.A.TrView()

		sol := NewPredCorr()
		bench.StartTimer()
		sol.Solve(mdl)
		bench.StopTimer()

		rd.Sub(mdl.C, mdl.S)
		rd.AddMul(At, mdl.Y, -1)
		rp.Apply(mdl.A, mdl.X)
		rp.Sub(mdl.B, rp)
		rs.Mul(mdl.X, mdl.S)
		rs.Neg(rs)
		dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
		if dev > tol {
			bench.Log(dev)
		}
	}
}
