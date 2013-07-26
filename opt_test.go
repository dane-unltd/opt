package opt

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

/*func TestLinprog2(t *testing.T) {
	m := 2
	n := 4
	tol := 1e-8

	A := mat.NewFromArray([]float64{1, 5, 2, 6, 3, 7, 4, 8}, true, m, n)
	At := A.TrView()
	b := mat.NewVec(m)
	b.AddSc(5)
	c := mat.NewVec(n)
	c.AddSc(-1)

	x, y, s := linprog(c, A, b, tol)

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	rd.Sub(c, s)
	rd.AddMul(At, y, 1)
	rp.Apply(A, x)
	rp.Sub(b, rp)
	rs.MulH(x, s)
	rs.Neg(rs)
	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if dev > tol {
		t.Log(dev)
		t.Fail()
	}
}*/

func TestLinprog(t *testing.T) {
	m := 5
	n := 10
	tol := 1e-8

	A := mat.RandN(m, n)
	At := A.TrView()
	b := mat.NewVec(m)
	c := mat.RandVec(n)
	xt := mat.RandVec(n)
	b.Apply(A, xt)

	x, y, s := linprog(c, A, b, tol)

	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	rd.Sub(c, s)
	rd.AddMul(At, y, -1)
	rp.Apply(A, x)
	rp.Sub(b, rp)
	rs.MulH(x, s)
	rs.Neg(rs)
	dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
	if dev > tol {
		t.Log(dev)
		t.Fail()
	}
}

func BenchmarkLinprog(bench *testing.B) {
	m := 500
	n := 1000
	tol := 1e-3
	rd := mat.NewVec(n)
	rp := mat.NewVec(m)
	rs := mat.NewVec(n)

	x := mat.NewVec(n)
	y := mat.NewVec(m)
	s := mat.NewVec(n)
	for i := 0; i < bench.N; i++ {
		bench.StopTimer()
		A := mat.RandN(m, n)
		At := A.TrView()
		b := mat.NewVec(m)
		c := mat.RandVec(n)
		xt := mat.RandVec(n)
		b.Apply(A, xt)

		bench.StartTimer()
		x, y, s = linprog(c, A, b, tol)
		bench.StopTimer()

		rd.Sub(c, s)
		rd.AddMul(At, y, -1)
		rp.Apply(A, x)
		rp.Sub(b, rp)
		rs.MulH(x, s)
		rs.Neg(rs)
		dev := (rd.Asum() + rp.Asum() + rs.Asum()) / float64(n)
		if dev > tol {
			bench.Log(dev)
		}
	}
}
